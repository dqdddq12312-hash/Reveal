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
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel, ValidationError

from openai import OpenAI

from esg_metadata import ESGMetadataModule, ESGMetadataRecord


REQUIRED_RESPONSE_FIELDS = [
    "Disclosure",
    "KPI",
    "Topic",
    "Value",
    "Unit",
    "Target",
    "Action",
]


class ESGAgentResponse(BaseModel):
    Disclosure: str
    KPI: str
    Topic: str
    Value: str
    Unit: str
    Target: str
    Action: str


def _normalize_disclosure(value: str) -> str:
    lookup = {
        "yes": "Yes",
        "no": "No",
        "partial": "Partial",
        "not available": "Not Available",
    }
    key = (value or "").strip().lower()
    return lookup.get(key, value if value else "Not Available")


def _extract_numeric_tokens(value: str) -> List[str]:
    if not value:
        return []
    return re.findall(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value)


def _ensure_schema_fields(payload: Any) -> Dict[str, str]:
    if not isinstance(payload, dict):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    result: Dict[str, str] = {}
    for field in REQUIRED_RESPONSE_FIELDS:
        raw_value = payload.get(field, "Not Available")
        result[field] = str(raw_value).strip() or "Not Available"
    result["Disclosure"] = _normalize_disclosure(result["Disclosure"])
    return result


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
        min_text_evidence: int = 1,
        min_table_evidence: int = 0,
        min_total_evidence: int = 2,
        allow_table_only: bool = True,
        table_override_min_tables: int = 1,
        table_override_min_score: float = 0.25,
        table_override_min_confidence: float = 0.55,
        consensus_runs: int = 1,
        consensus_temperature_delta: float = 0.15,
        debug_mode: bool = False,
    ) -> None:
        self.kb_root = Path(kb_root)
        self.model_name = model_name
        self.temperature = temperature
        self.top_k_per_source = top_k_per_source
        self.rerank_top_n = rerank_top_n
        self.min_text_evidence = max(0, min_text_evidence)
        self.min_table_evidence = max(0, min_table_evidence)
        self.min_total_evidence = max(1, min_total_evidence)
        self._base_min_text = self.min_text_evidence
        self._base_min_table = self.min_table_evidence
        self._base_min_total = self.min_total_evidence
        self.allow_table_only = allow_table_only
        self.table_override_min_tables = max(1, table_override_min_tables)
        self.table_override_min_score = table_override_min_score
        self.table_override_min_confidence = table_override_min_confidence
        self.consensus_runs = max(2, consensus_runs)
        self.consensus_temperature_delta = consensus_temperature_delta
        self.debug_mode = debug_mode
        self.metadata_tag_boost = 0.8
        self.metadata_dependency_boost = 0.4

        # Maintain per-source quotas so guardrails always see diverse evidence.
        self.source_minima = {"text": 1, "table": 1}
        outline_cap = max(1, self.top_k_per_source // 2)
        per_source_cap = max(1, self.top_k_per_source)
        self.source_maxima = {"text": per_source_cap, "table": per_source_cap, "outline": outline_cap}
        self.max_evidence = max(3, sum(self.source_maxima.values()))

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

    def _apply_metadata_boost(
        self,
        evidence: List[Dict[str, Any]],
        record: ESGMetadataRecord,
    ) -> None:
        indicator_id = record.indicator_id
        dependency_tags = set(record.dependency_groups or [])
        for ev in evidence:
            metadata = ev.get("metadata") or {}
            tags = set(metadata.get("tags", []))
            boost = 0.0
            if indicator_id and indicator_id in tags:
                boost += self.metadata_tag_boost
            if dependency_tags and dependency_tags.intersection(tags):
                boost += self.metadata_dependency_boost
            ev["metadata_boost"] = boost
            base_score = ev.get("rerank_score") if "rerank_score" in ev else ev.get("score", 0.0)
            ev["hybrid_score"] = base_score + boost

    def _select_evidence_with_quotas(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not evidence:
            return evidence
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for ev in evidence:
            grouped[ev.get("source_type", "unknown")].append(ev)
        for items in grouped.values():
            items.sort(key=lambda x: x.get("hybrid_score", x.get("score", 0.0)), reverse=True)

        selected: List[Dict[str, Any]] = []
        used_ids: set[int] = set()
        per_source_counts: Dict[str, int] = defaultdict(int)

        def _take(source: str, limit: int) -> None:
            if limit <= 0:
                return
            for ev in grouped.get(source, []):
                key = id(ev)
                if key in used_ids:
                    continue
                if self.source_maxima.get(source) and per_source_counts[source] >= self.source_maxima[source]:
                    return
                selected.append(ev)
                used_ids.add(key)
                per_source_counts[source] += 1
                if per_source_counts[source] >= limit:
                    return

        # Ensure minimum coverage for high-signal modalities when available.
        for source, minimum in self.source_minima.items():
            _take(source, minimum)

        all_ranked = sorted(
            evidence,
            key=lambda x: x.get("hybrid_score", x.get("score", 0.0)),
            reverse=True,
        )
        for ev in all_ranked:
            if len(selected) >= self.max_evidence:
                break
            key = id(ev)
            if key in used_ids:
                continue
            source = ev.get("source_type", "unknown")
            max_cap = self.source_maxima.get(source)
            if max_cap and per_source_counts[source] >= max_cap:
                continue
            selected.append(ev)
            used_ids.add(key)
            per_source_counts[source] += 1

        if not selected:
            # Fallback to highest scoring slice to avoid empty prompts.
            fallback_limit = max(1, min(self.top_k_per_source, len(evidence)))
            return all_ranked[:fallback_limit]
        return selected

    def _compute_evidence_metrics(
        self,
        evidence: List[Dict[str, Any]],
        record: ESGMetadataRecord,
    ) -> Dict[str, Any]:
        counts = Counter(ev.get("source_type", "unknown") for ev in evidence)
        total = sum(counts.values())
        requires_numeric = record.requires_numeric_evidence()
        required_text = 0 if requires_numeric else self._base_min_text
        required_table = max(self._base_min_table, 1 if requires_numeric else 0)
        required_total = max(self._base_min_total, required_table if requires_numeric else self._base_min_total)
        meets = (
            counts.get("text", 0) >= required_text
            and counts.get("table", 0) >= required_table
            and total >= required_total
        )
        reason = None
        reason_code = None
        override = None
        if not meets:
            if total == 0:
                reason_code = "no_evidence"
                reason = "No retrieval candidates passed the guardrail filters."
            else:
                reason_code = "insufficient_mix"
                reason = (
                    f"Needed text>={required_text}, table>={required_table}, total>={required_total}; "
                    f"got text={counts.get('text',0)}, table={counts.get('table',0)}, total={total}."
                )
            if self._table_override_ready(record, evidence, counts):
                meets = True
                reason = None
                reason_code = None
                override = "table_only"
        counts_dict = dict(counts)
        counts_dict.update(
            {
                "total": total,
                "meets_threshold": meets,
                "reason": reason,
                "reason_code": reason_code,
                "override": override,
                "requirements": {
                    "text": required_text,
                    "table": required_table,
                    "total": required_total,
                },
            }
        )
        return counts_dict

    def _table_override_ready(
        self,
        record: ESGMetadataRecord,
        evidence: List[Dict[str, Any]],
        counts: Dict[str, Any],
    ) -> bool:
        if not self.allow_table_only or not record.requires_numeric_evidence():
            return False
        if counts.get("text", 0) > 0:
            return False
        min_tables_needed = max(self.table_override_min_tables, self._base_min_table if not record.requires_numeric_evidence() else 1)
        if counts.get("table", 0) < min_tables_needed:
            return False
        indicator_id = record.indicator_id
        for ev in evidence:
            if ev.get("source_type") != "table":
                continue
            metadata = ev.get("metadata") or {}
            tags = metadata.get("tags") or []
            stats = metadata.get("stats") or {}
            confidence = float(stats.get("confidence", 0.0))
            score = float(ev.get("hybrid_score", ev.get("rerank_score", ev.get("score", 0.0))))
            table_text = (metadata.get("table_text") or "").lower()
            has_tag = bool(indicator_id and indicator_id in tags)
            has_unit_hint = bool(record.canonical_units and any((unit or "").lower() in table_text for unit in record.canonical_units))
            high_confidence = confidence >= self.table_override_min_confidence
            if (has_tag or has_unit_hint or high_confidence) and score >= self.table_override_min_score:
                return True
        return False

    def _build_guardrail_lines(self, record: ESGMetadataRecord) -> List[str]:
        lines = [
            "Use only the provided evidence snippets; do not speculate.",
            "Cite exact figures and wording from the evidence.",
        ]
        if record.requires_numeric_evidence():
            lines.append("Report exact numeric values and intensities; do not provide qualitative summaries only.")
        if record.canonical_units:
            unit_list = ", ".join(record.canonical_units)
            lines.append(f"Express values using the canonical units: {unit_list}.")
        if record.expected_time_series():
            lines.append(
                f"Include at least {record.expected_time_series()} consecutive reporting periods if available."
            )
        if record.requires_time_series():
            lines.append("Highlight year-by-year trends; specify the years shown in the evidence.")
        if record.evidence_type == "policy":
            lines.append("Focus on governance/process narratives; avoid inventing numeric results.")
        lines.append("If information is missing, output 'Not Available'.")
        return lines

    def _build_unavailable_response(
        self,
        record: ESGMetadataRecord,
        metrics: Dict[str, Any],
    ) -> Dict[str, str]:
        reason = metrics.get("reason") or "Evidence threshold not met"
        reason_code = metrics.get("reason_code") or "insufficient_evidence"
        disclosure = "Not Available" if reason_code == "no_evidence" else "Partial"
        return {
            "Disclosure": disclosure,
            "KPI": record.kpi,
            "Topic": ", ".join(record.topic),
            "Value": "Not Available",
            "Unit": "N/A",
            "Target": "Not Available",
            "Action": reason,
        }

    def _parse_and_validate(self, raw_answer: str) -> Tuple[Optional[Dict[str, str]], Dict[str, Any]]:
        log: Dict[str, Any] = {"status": "ok"}
        extracted = self._extract_json(raw_answer)
        if extracted is None:
            log["status"] = "json_not_found"
            return None, log
        try:
            model = ESGAgentResponse(**_ensure_schema_fields(extracted))
            return model.dict(), log
        except ValidationError as exc:
            log["status"] = "validation_error"
            log["errors"] = exc.errors()
            try:
                model = ESGAgentResponse(**_ensure_schema_fields(extracted))
                log["status"] = "auto_repaired"
                return model.dict(), log
            except ValidationError:
                return None, log

    def _calibrate_answer(
        self,
        parsed: Optional[Dict[str, str]],
        evidence: List[Dict[str, Any]],
        record: ESGMetadataRecord,
    ) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, Any]]]:
        if not parsed:
            return parsed, None
        calibration: Dict[str, Any] = {
            "numeric_match": True,
            "unit_match": True,
            "downgraded": False,
        }
        evidence_text = "\n".join(ev.get("content", "") for ev in evidence).lower()
        evidence_chunks = [ev.get("content", "") for ev in evidence]
        value_text = parsed.get("Value", "")
        numbers = _extract_numeric_tokens(value_text)
        reasons: List[str] = []
        disclosure = _normalize_disclosure(parsed.get("Disclosure", ""))

        if record.requires_numeric_evidence():
            if not numbers:
                calibration["numeric_match"] = False
                reasons.append("No numeric value extracted for quantitative KPI.")
            else:
                missing = []
                matched_numbers = []
                for num in numbers:
                    normalized = num.replace(",", "")
                    found = False
                    for chunk in evidence_chunks:
                        if normalized in chunk.replace(",", ""):
                            found = True
                            matched_numbers.append({"number": num, "chunk_preview": chunk[:160]})
                            break
                    if not found:
                        missing.append(num)
                calibration["matched_numbers"] = matched_numbers
                if missing:
                    calibration["numeric_match"] = False
                    reasons.append(f"Numbers not found in evidence: {', '.join(missing)}")

        if record.canonical_units:
            units_present = any(unit.lower() in value_text.lower() for unit in record.canonical_units)
            if not units_present:
                calibration["unit_match"] = False
                reasons.append("Canonical unit missing in Value field.")

        if not calibration["numeric_match"] and not numbers:
            parsed["Disclosure"] = "Not Available"
            calibration["downgraded"] = True
        elif (not calibration["numeric_match"] or not calibration["unit_match"]) and disclosure.lower() == "yes":
            parsed["Disclosure"] = "Partial"
            calibration["downgraded"] = True

        if reasons:
            calibration["reasons"] = reasons
        return parsed, calibration

    def _collect_answers(
        self,
        prompt: str,
        evidence: List[Dict[str, Any]],
        record: ESGMetadataRecord,
    ) -> Dict[str, Any]:
        runs = self.consensus_runs
        raw_answers: List[str] = []
        parsed_answers: List[Optional[Dict[str, str]]] = []
        validation_logs: List[Dict[str, Any]] = []
        calibration_logs: List[Optional[Dict[str, Any]]] = []
        for idx in range(runs):
            temp = self.temperature + (idx * self.consensus_temperature_delta if runs > 1 else 0)
            raw = self._call_llm(prompt, temperature=temp)
            raw_answers.append(raw)
            parsed, validation_log = self._parse_and_validate(raw)
            validation_logs.append(validation_log)
            calibrated, calibration_log = self._calibrate_answer(
                parsed.copy() if parsed else parsed,
                evidence,
                record,
            )
            parsed_answers.append(calibrated)
            calibration_logs.append(calibration_log)
        consensus = self._summarize_consensus(parsed_answers)
        return {
            "raw_answers": raw_answers,
            "parsed_answers": parsed_answers,
            "validation_logs": validation_logs,
            "calibration_logs": calibration_logs,
            "consensus": consensus,
        }

    def _summarize_consensus(self, parsed_answers: List[Optional[Dict[str, str]]]) -> Dict[str, Any]:
        valid = [ans for ans in parsed_answers if ans]
        if not valid:
            return {
                "final_answer": None,
                "agreement": 0.0,
                "runs": len(parsed_answers),
            }
        disclosures = [
            _normalize_disclosure(ans.get("Disclosure", ""))
            for ans in valid
        ]
        counter = Counter(disclosures)
        top_label, top_count = counter.most_common(1)[0]
        agreement = top_count / len(valid)
        final_answer = next(
            (ans for ans in valid if _normalize_disclosure(ans.get("Disclosure", "")) == top_label),
            valid[0],
        )
        final_answer = dict(final_answer)
        final_answer["Disclosure"] = top_label
        return {
            "final_answer": final_answer,
            "agreement": round(agreement, 3),
            "runs": len(parsed_answers),
            "distribution": counter,
        }

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
        kb_path: Optional[Path] = None
        legacy_base_name: Optional[str] = None
        legacy_mode = False

        if kb_dir.is_dir():
            kb_path = kb_dir / "knowledge_base.json"
            if not kb_path.exists():
                raise FileNotFoundError(f"knowledge_base.json not found in {kb_dir}")
        else:
            # Legacy layout: <kb_root>/<report>_kb.json plus *_text.index files
            legacy_candidates: List[Path] = []
            direct_candidate = self.kb_root / report_name
            if direct_candidate.is_file():
                legacy_candidates.append(direct_candidate)
            legacy_candidates.append(self.kb_root / f"{report_name}_kb.json")
            if report_name.endswith("_kb"):
                legacy_candidates.append(self.kb_root / f"{report_name}.json")
            for candidate in legacy_candidates:
                if candidate.exists():
                    kb_path = candidate
                    break
            if kb_path is None:
                raise FileNotFoundError(
                    "Could not find knowledge base directory or *_kb.json file for "
                    f"report '{report_name}' under {self.kb_root}."
                )
            legacy_mode = True
            legacy_base_name = kb_path.stem
            if legacy_base_name.endswith("_kb"):
                legacy_base_name = legacy_base_name[:-3]
            kb_dir = kb_path.parent

        with kb_path.open("r", encoding="utf-8") as f:
            kb_data = json.load(f)

        def _read_index(name: str) -> Optional[faiss.Index]:
            if legacy_mode:
                if not legacy_base_name:
                    return None
                idx_path = self.kb_root / f"{legacy_base_name}_{name}.index"
            else:
                idx_path = (self.kb_root / report_name) / f"{name}.index"
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
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
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
        metrics : dict
            Evidence counts, thresholds, and guardrail readiness signals.
        """
        kb = kb_handle.knowledge_base
        query_text = self._build_query_text(record)
        query_vec = self._encode_query(query_text)

        evidence: List[Dict[str, Any]] = []

        # --- Text summaries ---
        text_summaries = self._safe_get(kb, "text", "summaries", default=[]) or []
        text_originals = self._safe_get(kb, "text", "original_texts", default=[]) or []
        text_embs = self._safe_get(kb, "text", "embeddings", default=[]) or []
        text_metadata = self._safe_get(kb, "text", "metadata", default=[]) or []
        scores, ids = self._retrieve_from_index(
            query_vec, kb_handle.text_index, text_embs, self.top_k_per_source
        )
        for score, idx in zip(scores, ids):
            if idx < 0 or idx >= len(text_summaries):
                continue
            summary = text_summaries[idx]
            orig = text_originals[idx] if idx < len(text_originals) else ""
            metadata_entry = text_metadata[idx] if idx < len(text_metadata) else {}
            content = f"[Text block]\n{orig}\n\n[Summary]\n{summary}"
            evidence.append(
                {
                    "source_type": "text",
                    "score": float(score),
                    "content": content,
                    "extra": {"index": int(idx)},
                    "metadata": metadata_entry,
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
                    "metadata": {"level": level},
                }
            )

        # --- Table indicator phrases + full tables ---
        table_phrases = self._safe_get(kb, "tables", "phrases", default=[]) or []
        table_ids = self._safe_get(kb, "tables", "table_ids", default=[]) or []
        tables = self._safe_get(kb, "tables", "tables", default=[]) or []
        table_embs = self._safe_get(kb, "tables", "embeddings", default=[]) or []
        table_metadata = self._safe_get(kb, "tables", "metadata", default=[]) or []
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
            metadata_entry = table_metadata[t_id] if (table_metadata and t_id is not None and 0 <= t_id < len(table_metadata)) else {}
            content = f"[Table indicator] {phrase}\n{table_text}"
            evidence.append(
                {
                    "source_type": "table",
                    "score": float(score),
                    "content": content,
                    "extra": {"phrase_index": int(idx), "table_id": t_id},
                    "metadata": metadata_entry,
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
        self._apply_metadata_boost(evidence, record)
        evidence.sort(key=lambda x: x.get("hybrid_score", x.get("rerank_score", x.get("score", 0.0))), reverse=True)

        # Limit to top-N overall while preserving modality diversity
        top_evidence = self._select_evidence_with_quotas(evidence)
        metrics = self._compute_evidence_metrics(top_evidence, record)
        # Concatenate into a single reference content block
        reference_parts = []
        for i, ev in enumerate(top_evidence, start=1):
            reference_parts.append(f"### Evidence {i} ({ev['source_type']})\n{ev['content']}")
        reference_text = "\n\n".join(reference_parts)

        return reference_text, top_evidence, metrics

    # --------------------------- prompting & LLM ----------------------------

    def build_prompt(
        self,
        record: ESGMetadataRecord,
        kb_handle: ESGKnowledgeBaseHandle,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Build an ESGReveal-style prompt for a given ESG metadata record
        by retrieving relevant evidence and then delegating to the metadata module
        for prompt construction.

        Returns the final prompt text, the evidence list, and the evidence metrics
        used for guardrail enforcement.
        """
        reference_text, evidence, metrics = self.retrieve_evidence(record, kb_handle)
        base_prompt = self.metadata.build_prompt_from_metadata(record, reference_text)
        guardrails = "\n".join(f"- {line}" for line in self._build_guardrail_lines(record))
        provenance_lines = []
        for ev in evidence:
            meta = ev.get("metadata") or {}
            page = meta.get("page")
            source = ev.get("source_type", "text")
            if page is not None:
                provenance_lines.append(f"  - {source} evidence from page {page}")
        provenance_block = "\n".join(provenance_lines)
        prompt = f"{base_prompt}\n**Evidence Provenance:**\n{provenance_block or '  - Not available'}\n**Guardrails:**\n{guardrails}\n"
        return prompt, evidence, metrics

    def _call_llm(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Call the configured LLM (Chat Completions API) and return raw text.
        """
        temp = self.temperature if temperature is None else temperature
        system_prompt = (
            "You are an ESG data extraction agent (ESGReveal). "
            "You MUST answer strictly based on the provided reference content. "
            "If the information is not present, you MUST say it is not available. "
            "Always respond in JSON with fields: "
            '\"Disclosure\", \"KPI\", \"Topic\", \"Value\", \"Unit\", \"Target\", \"Action\".'
        )
        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temp,
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
                            "raw_answers": [str, ...],
                            "raw_answer": str | None,
                            "parsed_answer": dict | None,
                            "parsed_answers": [dict | None, ...],
                            "validation_logs": [...],
                            "calibration_logs": [...],
                            "consensus": {...},
                            "evidence_metrics": {...},
            }
        """
        kb_handle = self._load_kb(report_name)
        prompt, evidence, metrics = self.build_prompt(record, kb_handle)
        if not metrics.get("meets_threshold", True):
            fallback = self._build_unavailable_response(record, metrics)
            result = {
                "report": report_name,
                "metadata": record.to_json(),
                "prompt": prompt,
                "evidence": evidence,
                "raw_answers": [],
                "raw_answer": None,
                "parsed_answer": fallback,
                "parsed_answers": [],
                "validation_logs": [],
                "calibration_logs": [],
                "evidence_metrics": metrics,
                "consensus": {
                    "final_answer": fallback,
                    "agreement": 0.0,
                    "runs": 0,
                },
            }
            if self.debug_mode:
                result["debug"] = {
                    "evidence": evidence,
                    "settings": {
                        "min_text_evidence": self.min_text_evidence,
                        "min_table_evidence": self.min_table_evidence,
                        "min_total_evidence": self.min_total_evidence,
                        "consensus_runs": self.consensus_runs,
                    },
                }
            return result

        answer_bundle = self._collect_answers(prompt, evidence, record)
        final_answer = answer_bundle["consensus"].get("final_answer")
        result = {
            "report": report_name,
            "metadata": record.to_json(),
            "prompt": prompt,
            "evidence": evidence,
            "raw_answer": answer_bundle["raw_answers"][0] if answer_bundle["raw_answers"] else None,
            "raw_answers": answer_bundle["raw_answers"],
            "parsed_answer": final_answer,
            "parsed_answers": answer_bundle["parsed_answers"],
            "validation_logs": answer_bundle["validation_logs"],
            "calibration_logs": answer_bundle["calibration_logs"],
            "consensus": answer_bundle["consensus"],
            "evidence_metrics": metrics,
        }
        if self.debug_mode:
            result["debug"] = {
                "evidence": evidence,
                "settings": {
                    "min_text_evidence": self.min_text_evidence,
                    "min_table_evidence": self.min_table_evidence,
                    "min_total_evidence": self.min_total_evidence,
                    "consensus_runs": self.consensus_runs,
                },
            }
        return result


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
