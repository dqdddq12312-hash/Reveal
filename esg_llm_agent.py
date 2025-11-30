"""
esg_llm_agent.py

LLM Agent Module for ESGReveal-style ESG data extraction.

This module is designed to work together with:

- esg_metadata.ESGMetadataModule / ESGMetadataRecord
- esg_report_processing.ESGReportPreprocessor

It implements the "LLM agent module" described in the ESGReveal paper:
  * multi-type retrieval from text / outline / tables knowledge bases
  * vector similarity search using the same embedding model as preprocessing (M3E-base)
  * optional reranking of top candidates using a cross-encoder (coROM-style)
  * prompt construction based on ESG metadata (via build_prompt_from_metadata)
  * LLM answering that returns JSON with fields:
      <Disclosure, KPI, Topic, Value, Unit, Target, Action>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from openai import OpenAI

from esg_metadata import ESGMetadataModule, ESGMetadataRecord


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ESGKnowledgeBaseHandle:
    """In-memory handle for a single report's knowledge base + FAISS indexes."""
    name: str
    kb_dir: Path
    knowledge_base: Dict[str, Any]
    text_index: Optional[faiss.Index]
    outline_index: Optional[faiss.Index]
    table_index: Optional[faiss.Index]


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

class ESGLLMAgent:
    """
    ESGLLMAgent implements the ESGReveal-style LLM Agent Module.

    Responsibilities
    ----------------
    1. Load per-report knowledge bases and FAISS indexes produced by ESGReportPreprocessor.
    2. For a given ESGMetadataRecord, build a semantic query and retrieve the most
       relevant evidence from:
         - text summaries
         - document outline headers
         - table indicator phrases + tables
    3. (Optional) Re-rank retrieved evidence using a cross-encoder reranker
       (coROM-style, here approximated with a public reranker model).
    4. Build an ESGReveal prompt via ESGMetadataModule.build_prompt_from_metadata.
    5. Call an LLM (e.g., gpt-4o-mini) and parse JSON answers with fields
       {Disclosure, KPI, Topic, Value, Unit, Target, Action}.
    """

    def __init__(
        self,
        kb_root: str | Path,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_k_per_source: int = 4,
        rerank_top_n: int = 12,
        embedding_model_name: str = "moka-ai/m3e-base",
        reranker_model_name: str = "BAAI/bge-reranker-base",
        metadata_module: ESGMetadataModule | None = None,
    ) -> None:
        self.kb_root = Path(kb_root)
        self.model_name = model_name
        self.temperature = temperature
        self.top_k_per_source = top_k_per_source
        self.rerank_top_n = rerank_top_n

        # Metadata module (ESG metadata + prompt builder)
        self.metadata = metadata_module or ESGMetadataModule()

        # Embedding model, same as in report preprocessing module.
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Optional reranker (coROM-style). If loading fails, we simply skip reranking.
        try:
            self.reranker: Optional[CrossEncoder] = CrossEncoder(reranker_model_name)
        except Exception as exc:
            print(
                "Warning: Reranker model could not be loaded "
                f"({reranker_model_name}). Reranking will be skipped.\nError: {exc}"
            )
            self.reranker = None

        # OpenAI client (expects OPENAI_API_KEY in environment)
        self.client = OpenAI()

        # Cache for knowledge bases loaded from disk
        self._kb_cache: Dict[str, ESGKnowledgeBaseHandle] = {}

    # ---------------------------- utilities ---------------------------------

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return vecs / norms

    def _encode_query(self, text: str) -> np.ndarray:
        """Encode a query string into a normalized embedding vector."""
        emb = self.embedding_model.encode([text], show_progress_bar=False)
        return self._normalize(np.asarray(emb, dtype=np.float32))

    @staticmethod
    def _table_to_text(table: List[List[str]]) -> str:
        """Convert a table (rows x columns) into a readable text block."""
        lines = []
        for row in table:
            cells = [str(c).strip() for c in row]
            lines.append(" | ".join(cells))
        return "\n".join(lines)

    @staticmethod
    def _safe_get(d: Dict[str, Any], *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # --------------------------- KB loading ---------------------------------

    def _load_kb(self, report_name: str) -> ESGKnowledgeBaseHandle:
        """
        Load knowledge_base.json and FAISS indexes for a given report.

        Parameters
        ----------
        report_name : str
            Directory name under kb_root that contains:
              - knowledge_base.json
              - text.index, outline.index, table.index (optional)
        """
        if report_name in self._kb_cache:
            return self._kb_cache[report_name]

        kb_dir = self.kb_root / report_name
        if not kb_dir.exists():
            raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

        kb_path = kb_dir / "knowledge_base.json"
        if not kb_path.exists():
            raise FileNotFoundError(f"knowledge_base.json not found in {kb_dir}")

        with kb_path.open("r", encoding="utf-8") as f:
            kb_data = json.load(f)

        def _read_index(name: str) -> Optional[faiss.Index]:
            idx_path = kb_dir / f"{name}.index"
            if idx_path.exists():
                return faiss.read_index(str(idx_path))
            return None

        handle = ESGKnowledgeBaseHandle(
            name=report_name,
            kb_dir=kb_dir,
            knowledge_base=kb_data,
            text_index=_read_index("text"),
            outline_index=_read_index("outline"),
            table_index=_read_index("table"),
        )
        self._kb_cache[report_name] = handle
        return handle

    # --------------------------- retrieval ----------------------------------

    def _build_query_text(self, record: ESGMetadataRecord) -> str:
        """
        Construct a query string from ESG metadata, following ESGReveal's idea:
        Aspect + KPI + Topic + Quantity + SearchTerms.
        """
        topics_str = ", ".join(record.topic)
        search_terms_str = ", ".join(record.search_terms)
        query = (
            f"Aspect: {record.aspect}. "
            f"KPI: {record.kpi}. "
            f"Topics: {topics_str}. "
            f"Quantity: {record.quantity}. "
            f"Search terms: {search_terms_str}."
        )
        return query

    def _retrieve_from_index(
        self,
        query_vec: np.ndarray,
        index: Optional[faiss.Index],
        embeddings: List[List[float]] | np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper: retrieve top_k results from a FAISS index; if index is None,
        fall back to brute-force similarity over embeddings.
        """
        if index is not None:
            k = min(top_k, index.ntotal)
            if k == 0:
                return np.array([]), np.array([], dtype=int)
            scores, ids = index.search(query_vec, k)
            return scores[0], ids[0].astype(int)

        # Fallback: brute-force cosine similarity on CPU
        if embeddings is None:
            return np.array([]), np.array([], dtype=int)
        emb_array = np.asarray(embeddings, dtype=np.float32)
        if emb_array.size == 0:
            return np.array([]), np.array([], dtype=int)
        emb_array = self._normalize(emb_array)
        sims = np.dot(emb_array, query_vec[0])
        top_ids = np.argsort(sims)[-top_k:][::-1]
        return sims[top_ids], top_ids.astype(int)

    def retrieve_evidence(
        self,
        record: ESGMetadataRecord,
        kb_handle: ESGKnowledgeBaseHandle,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve multi-type evidence (text, outline, tables) for a metadata record
        and return a concatenated reference text plus per-evidence metadata.

        Returns
        -------
        reference_text : str
            The reference context passed into the ESGReveal-style prompt.
        evidence_list : list[dict]
            Each dict describes an evidence chunk with keys:
              - source_type: 'text' | 'outline' | 'table'
              - score: similarity score from vector search
              - content: human-readable text
              - extra: additional raw info (indices, etc.)
        """
        kb = kb_handle.knowledge_base
        query_text = self._build_query_text(record)
        query_vec = self._encode_query(query_text)

        evidence: List[Dict[str, Any]] = []

        # --- Text summaries ---
        text_summaries = self._safe_get(kb, "text", "summaries", default=[]) or []
        text_originals = self._safe_get(kb, "text", "original_texts", default=[]) or []
        text_embs = self._safe_get(kb, "text", "embeddings", default=[]) or []
        scores, ids = self._retrieve_from_index(
            query_vec, kb_handle.text_index, text_embs, self.top_k_per_source
        )
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(text_summaries):
                continue
            summary = text_summaries[idx]
            orig = text_originals[idx] if idx < len(text_originals) else ""
            content = f"[Text block]\n{orig}\n\n[Summary]\n{summary}"
            evidence.append(
                {
                    "source_type": "text",
                    "score": float(score),
                    "content": content,
                    "extra": {"index": int(idx)},
                }
            )

        # --- Outline headers ---
        outline_headers = self._safe_get(kb, "outline", "headers", default=[]) or []
        outline_levels = self._safe_get(kb, "outline", "levels", default=[]) or []
        outline_embs = self._safe_get(kb, "outline", "embeddings", default=[]) or []
        scores, ids = self._retrieve_from_index(
            query_vec, kb_handle.outline_index, outline_embs, self.top_k_per_source
        )
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(outline_headers):
                continue
            header = outline_headers[idx]
            level = outline_levels[idx] if idx < len(outline_levels) else None
            content = f"[Outline level {level}]\n{header}"
            evidence.append(
                {
                    "source_type": "outline",
                    "score": float(score),
                    "content": content,
                    "extra": {"index": int(idx), "level": level},
                }
            )

        # --- Table indicator phrases + full tables ---
        table_phrases = self._safe_get(kb, "tables", "phrases", default=[]) or []
        table_ids = self._safe_get(kb, "tables", "table_ids", default=[]) or []
        tables = self._safe_get(kb, "tables", "tables", default=[]) or []
        table_embs = self._safe_get(kb, "tables", "embeddings", default=[]) or []
        scores, ids = self._retrieve_from_index(
            query_vec, kb_handle.table_index, table_embs, self.top_k_per_source
        )
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(table_phrases):
                continue
            phrase = table_phrases[idx]
            t_id = table_ids[idx] if idx < len(table_ids) else None
            table = tables[t_id] if t_id is not None and 0 <= t_id < len(tables) else []
            table_text = self._table_to_text(table)
            content = f"[Table indicator] {phrase}\n{table_text}"
            evidence.append(
                {
                    "source_type": "table",
                    "score": float(score),
                    "content": content,
                    "extra": {"phrase_index": int(idx), "table_id": t_id},
                }
            )

        # --- Optional reranking (coROM-style) ---
        if self.reranker is not None and evidence:
            try:
                pairs = [(query_text, e["content"]) for e in evidence]
                rerank_scores = self.reranker.predict(pairs)
                for e, rs in zip(evidence, rerank_scores):
                    e["rerank_score"] = float(rs)
                evidence.sort(key=lambda x: x["rerank_score"], reverse=True)
            except Exception as exc:
                print("Warning: reranker.predict failed, falling back to vector scores.", exc)
                for e in evidence:
                    e["rerank_score"] = e["score"]
                evidence.sort(key=lambda x: x["score"], reverse=True)
        else:
            for e in evidence:
                e["rerank_score"] = e["score"]
            evidence.sort(key=lambda x: x["score"], reverse=True)

        # Limit to top-N overall
        top_evidence = evidence[: max(1, self.top_k_per_source)]
        # Concatenate into a single reference content block
        reference_parts = []
        for i, ev in enumerate(top_evidence, start=1):
            reference_parts.append(f"### Evidence {i} ({ev['source_type']})\n{ev['content']}")
        reference_text = "\n\n".join(reference_parts)

        return reference_text, top_evidence

    # --------------------------- prompting & LLM ----------------------------

    def build_prompt(
        self,
        record: ESGMetadataRecord,
        kb_handle: ESGKnowledgeBaseHandle,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build an ESGReveal-style prompt for a given ESG metadata record
        by retrieving relevant evidence and then delegating to the metadata module
        for prompt construction.
        """
        reference_text, evidence = self.retrieve_evidence(record, kb_handle)
        prompt = self.metadata.build_prompt_from_metadata(record, reference_text)
        return prompt, evidence

    def _call_llm(self, prompt: str) -> str:
        """
        Call the configured LLM (Chat Completions API) and return raw text.
        """
        system_prompt = (
            "You are an ESG data extraction agent (ESGReveal). "
            "You MUST answer strictly based on the provided reference content. "
            "If the information is not present, you MUST say it is not available. "
            "Always respond in JSON with fields: "
            '\"Disclosure\", \"KPI\", \"Topic\", \"Value\", \"Unit\", \"Target\", \"Action\".'
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _extract_json(text: str) -> Any:
        """
        Try to extract a JSON object from an LLM response.
        ESGReveal expects the answer itself to be JSON, but we are defensive here:
        we search for the first {...} block and parse that.
        """
        text = text.strip()
        # Fast path: try parsing whole string
        try:
            return json.loads(text)
        except Exception:
            pass

        # Fallback: try to locate JSON object within text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None

    # --------------------------- public API ---------------------------------

    def answer_indicator(
        self,
        report_name: str,
        record: ESGMetadataRecord,
    ) -> Dict[str, Any]:
        """
        High-level API to run the full ESGReveal LLM Agent pipeline for one indicator.

        Steps:
          1. Load knowledge base + indexes for the given report.
          2. Retrieve multi-type evidence based on the metadata record.
          3. Build ESGReveal-style prompt.
          4. Call the LLM and parse JSON output.

        Returns
        -------
        result : dict
            {
              "report": report_name,
              "metadata": record.to_json(),
              "prompt": prompt,
              "evidence": [...],
              "raw_answer": str,
              "parsed_answer": dict | None,
            }
        """
        kb_handle = self._load_kb(report_name)
        prompt, evidence = self.build_prompt(record, kb_handle)
        raw_answer = self._call_llm(prompt)
        parsed = self._extract_json(raw_answer)
        return {
            "report": report_name,
            "metadata": record.to_json(),
            "prompt": prompt,
            "evidence": evidence,
            "raw_answer": raw_answer,
            "parsed_answer": parsed,
        }


class ESGRevealAgent:
    """Legacy in-memory agent used by run_esg_llm_agent.py."""

    def __init__(
        self,
        knowledge_base: Dict[str, Any],
        faiss_indexes: Dict[str, Optional[faiss.Index]],
        use_corom: bool = True,
        llm_model_name: str = "gpt-4o-mini",
        temperature: float | None = None,
        embedding_model_name: str = "moka-ai/m3e-base",
        reranker_model_name: str = "BAAI/bge-reranker-base",
        metadata_module: ESGMetadataModule | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.text_index = faiss_indexes.get("text_index")
        self.outline_index = faiss_indexes.get("outline_index")
        self.table_index = faiss_indexes.get("table_index")
        self.metadata_module = metadata_module or ESGMetadataModule()
        self.llm_model_name = llm_model_name
        self.default_temperature = 0.0 if temperature is None else temperature

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.reranker: Optional[CrossEncoder]
        if use_corom:
            try:
                self.reranker = CrossEncoder(reranker_model_name)
            except Exception as exc:
                print(
                    "Warning: Reranker model could not be loaded "
                    f"({reranker_model_name}). Reranking will be skipped.\nError: {exc}"
                )
                self.reranker = None
        else:
            self.reranker = None

        self.client = OpenAI()

    # --- shared helpers ---

    @staticmethod
    def _normalize(vecs: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return vecs / norms

    def _encode_query(self, text: str) -> np.ndarray:
        emb = self.embedding_model.encode([text], show_progress_bar=False)
        return self._normalize(np.asarray(emb, dtype=np.float32))

    @staticmethod
    def _table_to_text(table: List[List[str]]) -> str:
        lines = []
        for row in table:
            cells = [str(c).strip() for c in row]
            lines.append(" | ".join(cells))
        return "\n".join(lines)

    @staticmethod
    def _safe_get(d: Dict[str, Any], *keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _retrieve_from_index(
        self,
        query_vec: np.ndarray,
        index: Optional[faiss.Index],
        embeddings: List[List[float]] | np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if index is not None:
            k = min(top_k, index.ntotal)
            if k == 0:
                return np.array([]), np.array([], dtype=int)
            scores, ids = index.search(query_vec, k)
            return scores[0], ids[0].astype(int)

        if embeddings is None:
            return np.array([]), np.array([], dtype=int)
        emb_array = np.asarray(embeddings, dtype=np.float32)
        if emb_array.size == 0:
            return np.array([]), np.array([], dtype=int)
        emb_array = self._normalize(emb_array)
        sims = np.dot(emb_array, query_vec[0])
        top_ids = np.argsort(sims)[-top_k:][::-1]
        return sims[top_ids], top_ids.astype(int)

    # --- pipeline steps ---

    def retrieve_evidence(self, record: ESGMetadataRecord, top_k_per_source: int = 4):
        kb = self.knowledge_base
        query_text = (
            f"Aspect: {record.aspect}. "
            f"KPI: {record.kpi}. "
            f"Topics: {', '.join(record.topic)}. "
            f"Quantity: {record.quantity}. "
            f"Search terms: {', '.join(record.search_terms)}."
        )
        query_vec = self._encode_query(query_text)

        evidence: List[Dict[str, Any]] = []

        text_summaries = self._safe_get(kb, "text", "summaries", default=[])
        if text_summaries is None:
            text_summaries = []
        text_originals = self._safe_get(kb, "text", "original_texts", default=[])
        if text_originals is None:
            text_originals = []
        text_embs = self._safe_get(kb, "text", "embeddings", default=[])
        if text_embs is None:
            text_embs = []
        scores, ids = self._retrieve_from_index(
            query_vec, self.text_index, text_embs, top_k_per_source
        )
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(text_summaries):
                continue
            summary = text_summaries[idx]
            orig = text_originals[idx] if idx < len(text_originals) else ""
            content = f"[Text block]\n{orig}\n\n[Summary]\n{summary}"
            evidence.append(
                {
                    "source_type": "text",
                    "score": float(score),
                    "content": content,
                    "extra": {"index": int(idx)},
                }
            )

        outline_headers = self._safe_get(kb, "outline", "headers", default=[])
        if outline_headers is None:
            outline_headers = []
        outline_levels = self._safe_get(kb, "outline", "levels", default=[])
        if outline_levels is None:
            outline_levels = []
        outline_embs = self._safe_get(kb, "outline", "embeddings", default=[])
        if outline_embs is None:
            outline_embs = []
        scores, ids = self._retrieve_from_index(
            query_vec, self.outline_index, outline_embs, top_k_per_source
        )
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(outline_headers):
                continue
            header = outline_headers[idx]
            level = outline_levels[idx] if idx < len(outline_levels) else None
            content = f"[Outline level {level}]\n{header}"
            evidence.append(
                {
                    "source_type": "outline",
                    "score": float(score),
                    "content": content,
                    "extra": {"index": int(idx), "level": level},
                }
            )

        table_phrases = self._safe_get(kb, "tables", "phrases", default=[])
        if table_phrases is None:
            table_phrases = []
        table_ids = self._safe_get(kb, "tables", "table_ids", default=[])
        if table_ids is None:
            table_ids = []
        tables = self._safe_get(kb, "tables", "tables", default=[])
        if tables is None:
            tables = []
        table_embs = self._safe_get(kb, "tables", "embeddings", default=[])
        if table_embs is None:
            table_embs = []
        scores, ids = self._retrieve_from_index(
            query_vec, self.table_index, table_embs, top_k_per_source
        )
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(table_phrases):
                continue
            phrase = table_phrases[idx]
            t_id = table_ids[idx] if idx < len(table_ids) else None
            table = tables[t_id] if t_id is not None and 0 <= t_id < len(tables) else []
            table_text = self._table_to_text(table)
            content = f"[Table indicator] {phrase}\n{table_text}"
            evidence.append(
                {
                    "source_type": "table",
                    "score": float(score),
                    "content": content,
                    "extra": {"phrase_index": int(idx), "table_id": t_id},
                }
            )

        if self.reranker is not None and evidence:
            try:
                pairs = [(query_text, e["content"]) for e in evidence]
                rerank_scores = self.reranker.predict(pairs)
                for e, rs in zip(evidence, rerank_scores):
                    e["rerank_score"] = float(rs)
                evidence.sort(key=lambda x: x["rerank_score"], reverse=True)
            except Exception as exc:
                print("Warning: reranker.predict failed, falling back to vector scores.", exc)
                for e in evidence:
                    e["rerank_score"] = e["score"]
                evidence.sort(key=lambda x: x["score"], reverse=True)
        else:
            for e in evidence:
                e["rerank_score"] = e["score"]
            evidence.sort(key=lambda x: x["score"], reverse=True)

        top_evidence = evidence[: max(1, top_k_per_source)]
        reference_parts = []
        for i, ev in enumerate(top_evidence, start=1):
            reference_parts.append(f"### Evidence {i} ({ev['source_type']})\n{ev['content']}")
        reference_text = "\n\n".join(reference_parts)
        return reference_text, top_evidence

    def build_prompt(self, record: ESGMetadataRecord) -> Tuple[str, List[Dict[str, Any]]]:
        reference_text, evidence = self.retrieve_evidence(record)
        prompt = self.metadata_module.build_prompt_from_metadata(record, reference_text)
        return prompt, evidence

    def _call_llm(self, prompt: str, temperature: float) -> str:
        system_prompt = (
            "You are an ESG data extraction agent (ESGReveal). "
            "You MUST answer strictly based on the provided reference content. "
            "If the information is not present, you MUST say it is not available. "
            "Always respond in JSON with fields: "
            '"Disclosure", "KPI", "Topic", "Value", "Unit", "Target", "Action".'
        )
        resp = self.client.chat.completions.create(
            model=self.llm_model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _extract_json(text: str) -> Any:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None

    def answer_indicator(self, record: ESGMetadataRecord, temperature: float | None = None) -> Dict[str, Any]:
        temp = self.default_temperature if temperature is None else temperature
        prompt, evidence = self.build_prompt(record)
        raw_answer = self._call_llm(prompt, temperature=temp)
        parsed = self._extract_json(raw_answer)
        return {
            "record": record,
            "metadata": record.to_json(),
            "prompt": prompt,
            "evidence": evidence,
            "raw_answer": raw_answer,
            "parsed_answer": parsed,
        }


# ---------------------------------------------------------------------------
# CLI usage example
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="ESGReveal-style LLM Agent for ESG indicator extraction"
    )
    parser.add_argument(
        "--kb_root",
        type=str,
        default="processed_reports",
        help="Root directory where ESGReportPreprocessor saved knowledge bases.",
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help=(
            "Report name (directory under kb_root). For example, if your PDF was "
            "'BudweiserAPAC_2022_ESG.pdf', and you preprocessed it with "
            "esg_report_processing.py, use 'BudweiserAPAC_2022_ESG'."
        ),
    )
    parser.add_argument(
        "--aspect",
        type=str,
        default=None,
        help="Aspect filter (e.g., 'A1. Emissions').",
    )
    parser.add_argument(
        "--topic_query",
        type=str,
        default=None,
        help="Keyword to search in ESG metadata topics/search_terms (e.g., 'greenhouse gas').",
    )
    parser.add_argument(
        "--metadata_index",
        type=int,
        default=0,
        help="Index of the matched metadata record to use (0-based).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model name (e.g., gpt-4o-mini, gpt-4.1-mini, etc.).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the LLM.",
    )
    args = parser.parse_args()

    metadata_module = ESGMetadataModule()
    # Select metadata record
    candidates: List[ESGMetadataRecord] = []

    if args.aspect:
        candidates.extend(metadata_module.list_by_aspect(args.aspect))

    if args.topic_query:
        # Merge, avoiding duplicates
        search_results = metadata_module.search_by_topic(args.topic_query)
        for rec in search_results:
            if rec not in candidates:
                candidates.append(rec)

    if not candidates:
        print("No metadata records matched. Try adjusting --aspect or --topic_query.")
        raise SystemExit(1)

    if args.metadata_index < 0 or args.metadata_index >= len(candidates):
        print(f"--metadata_index out of range. Found {len(candidates)} candidates.")
        raise SystemExit(1)

    record = candidates[args.metadata_index]
    print("Using ESG metadata record:")
    print(json.dumps(record.to_json(), ensure_ascii=False, indent=2))

    agent = ESGLLMAgent(
        kb_root=args.kb_root,
        model_name=args.model,
        temperature=args.temperature,
        metadata_module=metadata_module,
    )

    result = agent.answer_indicator(args.report, record)
    print("\n--- Prompt sent to LLM ---\n")
    print(result["prompt"])
    print("\n--- Raw LLM answer ---\n")
    print(result["raw_answer"])
    print("\n--- Parsed JSON answer ---\n")
    print(json.dumps(result["parsed_answer"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
