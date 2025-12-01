from pathlib import Path
import argparse
import json
import re
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional, Iterable
from collections import Counter, defaultdict

import fitz  # PyMuPDF
import numpy as np
import faiss

from transformers import (
    AutoProcessor,
    AutoModelForObjectDetection,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from sentence_transformers import SentenceTransformer

from esg_metadata import ESGMetadataModule, ESGMetadataRecord


logger = logging.getLogger(__name__)


class ESGReportPreprocessor:
    """
    Report preprocessing module following ESGReveal:
    - parse_pdf: layout & table extraction (paragraphs, outline, tables)
    - build_knowledge_base: mT5 summaries + m3e embeddings for text/outline/tables
    - store_faiss_indexes: build FAISS indices for fast retrieval
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        offline_mode: bool = False,
        metadata_module: Optional[ESGMetadataModule] = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.offline_mode = offline_mode
        self.metadata_module = metadata_module or ESGMetadataModule()
        self._metadata_records: List[ESGMetadataRecord] = list(
            self.metadata_module._all_records()
        )
        self._metadata_by_id: Dict[str, ESGMetadataRecord] = {
            rec.indicator_id: rec
            for rec in self._metadata_records
            if rec.indicator_id
        }
        self._event_counters: Counter = Counter()
        self._processing_report: Dict[str, Any] = {}
        self._reset_processing_state()
        # Table detection & structure recognition (Table-Transformer family)
        self.table_detection_model_name = "microsoft/table-transformer-detection"
        self.table_structure_model_name = "microsoft/table-transformer-structure-recognition"

        try:
            self.detection_processor = AutoProcessor.from_pretrained(
                self.table_detection_model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.offline_mode,
            )
            self.detection_model = AutoModelForObjectDetection.from_pretrained(
                self.table_detection_model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.offline_mode,
            )
        except Exception as e:
            # Allow script to run without remote models (e.g. offline); table detection will be skipped.
            print(
                f"[WARN] Could not load table detection model '{self.table_detection_model_name}': {e}"
            )
            self.detection_processor = None
            self.detection_model = None

        try:
            self.structure_processor = AutoProcessor.from_pretrained(
                self.table_structure_model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.offline_mode,
            )
            self.structure_model = AutoModelForObjectDetection.from_pretrained(
                self.table_structure_model_name,
                cache_dir=self.cache_dir,
                local_files_only=self.offline_mode,
            )
        except Exception as e:
            print(
                f"[WARN] Could not load table structure model '{self.table_structure_model_name}': {e}"
            )
            self.structure_processor = None
            self.structure_model = None

        # Text summarisation (mT5-base)
        try:
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained(
                "google/mt5-base",
                cache_dir=self.cache_dir,
                local_files_only=self.offline_mode,
            )
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/mt5-base",
                cache_dir=self.cache_dir,
                local_files_only=self.offline_mode,
            )
        except Exception as e:
            print(f"[WARN] Could not load summariser (mt5-base): {e}")
            self.summarizer_tokenizer = None
            self.summarizer_model = None

        # Text embedding (m3e-base)
        try:
            self.embedding_model = SentenceTransformer(
                "moka-ai/m3e-base",
                cache_folder=self.cache_dir,
            )
        except Exception as e:
            print(f"[WARN] Could not load embedding model (m3e-base): {e}")
            self.embedding_model = None

    # ------------------------------------------------------------------
    # Internal helpers for provenance + tagging
    # ------------------------------------------------------------------

    def _reset_processing_state(self) -> None:
        self._event_counters.clear()
        self._processing_report = {}

    def _hash_content(self, text: str, page_index: int) -> str:
        payload = f"{page_index}:{text}".encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()

    def _compute_kb_digest(
        self,
        text_chunks: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
    ) -> str:
        hasher = hashlib.sha1()
        for chunk in text_chunks:
            hasher.update(chunk.get("hash", "").encode("utf-8", errors="ignore"))
        for table in tables:
            stats_blob = json.dumps(table.get("stats", {}), sort_keys=True)
            hasher.update(stats_blob.encode("utf-8", errors="ignore"))
            hasher.update(
                (table.get("table_text") or "").encode("utf-8", errors="ignore")
            )
        return hasher.hexdigest()

    def _metadata_fallback_summary(self, tags: List[str]) -> Optional[str]:
        for tag in tags or []:
            record = self._metadata_by_id.get(tag)
            if not record:
                continue
            knowledge = (record.knowledge or record.kpi or "").strip()
            if not knowledge:
                continue
            snippet = re.sub(r"\s+", " ", knowledge)
            if len(snippet) > 420:
                snippet = snippet[:417] + "..."
            return f"{record.indicator_id}: {snippet}"
        return None

    @staticmethod
    def _blocks_to_pseudo_rows(blocks: List[str]) -> List[List[str]]:
        rows: List[List[str]] = []
        for block in blocks:
            # Split on strong whitespace or pipes to approximate table columns
            parts = [
                cell.strip()
                for cell in re.split(r"\s{2,}|\||\t|\n", block)
                if cell.strip()
            ]
            if len(parts) >= 2:
                rows.append(parts[:8])
        return rows

    def _build_kb_metadata(
        self,
        source_pdf: Optional[str],
        text_chunks: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        schema_info = self.metadata_module.get_schema_info()
        pseudo_tables = sum(1 for table in tables if table.get("source") == "pseudo_table")
        return {
            "source_pdf": source_pdf,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": schema_info.get("version"),
            "schema_changelog": schema_info.get("changelog"),
            "metadata_records": len(self._metadata_records),
            "processing_report": self._processing_report,
            "content_hash": self._compute_kb_digest(text_chunks, tables),
            "pseudo_table_count": pseudo_tables,
        }

    @staticmethod
    def _clean_bbox(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
        if not bbox:
            return None
        x0, y0, x1, y1 = bbox
        return (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))

    def _split_block_text(
        self,
        text: str,
        max_chars: int = 900,
        overlap: int = 120,
    ) -> List[str]:
        """Split long paragraphs into overlapping windows to stabilise embeddings."""
        text = text.strip()
        if len(text) <= max_chars:
            return [text]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        buffer = ""
        for sentence in sentences:
            proposal = (buffer + " " + sentence).strip()
            if len(proposal) <= max_chars:
                buffer = proposal
                continue
            if buffer:
                chunks.append(buffer.strip())
                if overlap > 0 and len(buffer) > overlap:
                    buffer = buffer[-overlap:] + " " + sentence
                else:
                    buffer = sentence
            else:
                chunks.append(sentence[:max_chars])
                buffer = sentence[max_chars - overlap :]
        if buffer:
            chunks.append(buffer.strip())
        return [c for c in chunks if c]

    def _tag_text_with_metadata(self, text: str, max_matches: int = 16) -> List[str]:
        lower_text = text.lower()
        tags: List[str] = []
        seen: set[str] = set()
        for record in self._metadata_records:
            indicator_id = record.indicator_id
            if not indicator_id or indicator_id in seen:
                continue
            matched = any((term or "").lower() in lower_text for term in record.search_terms if term)
            if not matched and record.multilingual_synonyms:
                matched = any((syn or "").lower() in lower_text for syn in record.multilingual_synonyms if syn)
            if not matched and record.regex_hints:
                for pattern in record.regex_hints:
                    try:
                        if re.search(pattern, text, flags=re.IGNORECASE):
                            matched = True
                            break
                    except re.error:
                        continue
            if not matched and record.canonical_units:
                matched = any(
                    (unit or "").lower() in lower_text
                    for unit in record.canonical_units
                    if unit and len(unit) >= 2
                )
            if matched:
                tags.append(indicator_id)
                seen.add(indicator_id)
            if len(tags) >= max_matches:
                break
        return tags

    def _build_chunk_payload(
        self,
        chunk_text: str,
        page_index: int,
        bbox: Optional[Tuple[float, float, float, float]],
        heading: Optional[str],
    ) -> Dict[str, Any]:
        chunk_hash = self._hash_content(chunk_text, page_index)
        return {
            "text": chunk_text,
            "page": page_index,
            "bbox": self._clean_bbox(bbox),
            "heading": heading or "Document",
            "chunk_id": f"p{page_index}-{chunk_hash[:8]}",
            "hash": chunk_hash,
            "tags": self._tag_text_with_metadata(chunk_text),
        }

    def _tag_table_with_metadata(
        self,
        indicator_phrase: str,
        table_text: str,
        max_matches: int = 5,
    ) -> List[str]:
        combined = f"{indicator_phrase}\n{table_text}"
        return self._tag_text_with_metadata(combined, max_matches=max_matches)

    def _extract_numeric_values(self, table: List[List[str]]) -> List[float]:
        values: List[float] = []
        for row in table:
            for cell in row[1:]:
                if not isinstance(cell, str):
                    continue
                matches = re.findall(r"[+-]?\d+(?:\.\d+)?", cell.replace(",", ""))
                for match in matches:
                    try:
                        values.append(float(match))
                    except ValueError:
                        continue
        return values

    def _analyze_table(self, table: List[List[str]]) -> Dict[str, Any]:
        if not table:
            return {
                "year_columns": [],
                "numeric_ratio": 0.0,
                "min": None,
                "max": None,
                "suspect": True,
                "confidence": 0.0,
            }
        header = table[0] if table else []
        year_columns = [idx for idx, cell in enumerate(header) if self._is_year_header(cell)]
        total_cells = max(sum(len(row) for row in table), 1)
        numeric_cells = sum(
            1
            for row in table
            for cell in row
            if isinstance(cell, str) and re.search(r"\d", cell)
        )
        numeric_ratio = numeric_cells / total_cells
        numeric_values = self._extract_numeric_values(table)
        min_val = min(numeric_values) if numeric_values else None
        max_val = max(numeric_values) if numeric_values else None
        suspect = numeric_ratio < 0.15 or (year_columns and len(year_columns) < 2)
        confidence = min(0.95, max(0.05, numeric_ratio * 1.2))
        return {
            "year_columns": year_columns,
            "numeric_ratio": round(numeric_ratio, 3),
            "min": min_val,
            "max": max_val,
            "suspect": suspect,
            "confidence": round(confidence, 3),
        }

    def _build_table_payload(
        self,
        rows: List[List[str]],
        page_index: int,
        bbox: Optional[Tuple[float, float, float, float]],
        indicator_phrase: Optional[str] = None,
        page_level_tags: Optional[Iterable[str]] = None,
        source_label: str = "detected_table",
    ) -> Dict[str, Any]:
        stats = self._analyze_table(rows)
        if stats.get("suspect"):
            self._event_counters["tables_suspect"] += 1
        if source_label == "pseudo_table":
            stats["pseudo_table"] = True
            stats["confidence"] = round(min(stats.get("confidence", 0.4), 0.6), 3)
        table_text = self._table_to_text(rows)
        tags = set(self._tag_table_with_metadata(indicator_phrase or "", table_text))
        if page_level_tags:
            for tag in page_level_tags:
                if tag:
                    tags.add(tag)
        payload = {
            "rows": rows,
            "page": page_index,
            "bbox": self._clean_bbox(bbox),
            "stats": stats,
            "tags": sorted(tags),
            "table_text": table_text,
            "source": source_label,
        }
        return payload

    @staticmethod
    def _table_to_text(table: List[List[str]]) -> str:
        lines: List[str] = []
        for row in table:
            if not row:
                continue
            lines.append(" | ".join(cell.strip() for cell in row))
        return "\n".join(lines)

    def get_processing_report(self) -> Dict[str, Any]:
        return self._processing_report

    # -------------------------------------------------------------------------
    # Core PDF parsing
    # -------------------------------------------------------------------------

    def _detect_table_regions(
        self, doc: fitz.Document
    ) -> Dict[int, List[Tuple[float, float, float, float]]]:
        """
        Use Table-Transformer detection model to locate table bounding boxes (per page).
        Returns mapping: page_index -> list of (x0, y0, x1, y1) in page coordinate space.
        """
        table_regions_per_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
        if self.detection_model is None or self.detection_processor is None:
            return table_regions_per_page

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            # Render page to image (identity matrix so pixel coords ~= page coords)
            pix = page.get_pixmap()
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            inputs = self.detection_processor(images=img_array, return_tensors="pt")
            outputs = self.detection_model(**inputs)

            # HuggingFace DETR processors typically expose post_process_object_detection
            try:
                results = self.detection_processor.post_process_object_detection(
                    outputs,
                    threshold=0.5,
                    target_sizes=[(pix.height, pix.width)],
                )
            except Exception:
                try:
                    results = self.detection_processor.post_process(
                        outputs, target_sizes=[(pix.height, pix.width)]
                    )
                except Exception:
                    results = []

            table_regions: List[Tuple[float, float, float, float]] = []
            if results:
                page_result = results[0]
                # Modern processors: dict with tensors
                if isinstance(page_result, dict):
                    labels = page_result.get("labels", [])
                    boxes = page_result.get("boxes", [])
                    scores = page_result.get("scores", [])
                    for lbl, box, score in zip(labels, boxes, scores):
                        try:
                            label_idx = int(lbl.item())
                        except Exception:
                            label_idx = int(lbl)
                        label_name = (
                            self.detection_model.config.id2label.get(label_idx, "")
                            .lower()
                        )
                        try:
                            score_val = float(score.item())
                        except Exception:
                            score_val = float(score)
                        if label_name == "table" and score_val >= 0.5:
                            box_vals = box.tolist() if hasattr(box, "tolist") else box
                            if box_vals and len(box_vals) == 4:
                                x0, y0, x1, y1 = box_vals
                                table_regions.append((x0, y0, x1, y1))
                else:
                    # Older / alternative structure: list of dicts
                    for res in page_result:
                        if not isinstance(res, dict):
                            continue
                        label = str(res.get("label", "")).lower()
                        score = float(res.get("score", 0.0))
                        bbox = res.get("box", None)
                        if label == "table" and bbox and score >= 0.5:
                            table_regions.append(
                                (
                                    float(bbox["xmin"]),
                                    float(bbox["ymin"]),
                                    float(bbox["xmax"]),
                                    float(bbox["ymax"]),
                                )
                            )
            table_regions_per_page[page_index] = table_regions
        return table_regions_per_page

    def _extract_table_with_fitz(
        self, page: fitz.Page, clip_rect: fitz.Rect
    ) -> Optional[List[List[str]]]:
        """Fallback table extraction using PyMuPDF's built-in table finder."""

        try:
            finder = page.find_tables(clip=clip_rect)
        except Exception as exc:  # pragma: no cover - best-effort helper
            print(f"[WARN] PyMuPDF table fallback failed: {exc}")
            return None

        tables = list(getattr(finder, "tables", []))
        if not tables:
            try:
                finder = page.find_tables()
                tables = list(getattr(finder, "tables", []))
            except Exception:
                tables = []

        best_table = None
        best_overlap = 0.0
        clip_area = max(clip_rect.get_area(), 1e-6)

        for table in tables:
            try:
                table_rect = fitz.Rect(table.bbox)
            except Exception:
                continue
            intersection = clip_rect & table_rect
            if intersection.is_empty:
                continue
            overlap_ratio = intersection.get_area() / clip_area
            if overlap_ratio > best_overlap:
                best_table = table
                best_overlap = overlap_ratio

        if best_table is None:
            return None

        extracted_rows = []
        for row in best_table.extract():
            normalised_row: List[str] = []
            for cell in row:
                if cell is None:
                    normalised_row.append("")
                    continue
                cell_text = " ".join(str(cell).split())
                normalised_row.append(cell_text)
            if any(col for col in normalised_row):
                extracted_rows.append(normalised_row)

        return extracted_rows or None

    @staticmethod
    def _is_year_header(cell: Any) -> bool:
        if not isinstance(cell, str):
            return False
        token = cell.strip()
        return bool(re.fullmatch(r"(19|20)\d{2}", token))

    @staticmethod
    def _extract_numeric_tokens(cell: Any) -> List[str]:
        if not isinstance(cell, str):
            return []
        tokens = re.findall(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", cell)
        return [tok.strip() for tok in tokens if tok.strip()]

    def _expand_year_columns(self, table: List[List[str]]) -> List[List[str]]:
        header_idx = None
        year_cols: List[int] = []

        for idx, row in enumerate(table):
            current_year_cols = [col for col, cell in enumerate(row) if self._is_year_header(cell)]
            if len(current_year_cols) >= 3:  # need at least 3 years to treat as timeline
                header_idx = idx
                year_cols = current_year_cols
                break

        if header_idx is None or not year_cols:
            return table

        first_year_col = year_cols[0]
        num_years = len(year_cols)
        max_cols = max(len(row) for row in table)

        for row in table:
            if len(row) < max_cols:
                row.extend([""] * (max_cols - len(row)))

        for row_idx, row in enumerate(table):
            if row_idx <= header_idx:
                continue

            trailing_cells = row[first_year_col:]
            tokens: List[str] = []
            for cell in trailing_cells:
                tokens.extend(self._extract_numeric_tokens(cell))

            if len(tokens) >= num_years:
                for offset, col_idx in enumerate(year_cols):
                    if offset < len(tokens):
                        row[col_idx] = tokens[offset]

        return table

    def parse_pdf(
        self, pdf_path: str
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str]], List[Dict[str, Any]]]:
        """
        Parse the PDF and perform layout extraction.
        Returns:
            text_chunks: list of chunk dicts (text, page, bbox, heading, chunk_id, hash, tags)
            outline: list of (level, heading_text) entries
            tables: list of table payloads with rows, provenance, stats and tags
        """
        self._reset_processing_state()
        doc = fitz.open(pdf_path)
        self._event_counters["pages_processed"] += doc.page_count

        def _count_non_empty_cells(rows: List[List[str]]) -> int:
            return sum(
                1
                for row in rows
                for cell in row
                if isinstance(cell, str) and cell.strip()
            )

        def _count_numeric_cells(rows: List[List[str]]) -> int:
            return sum(
                1
                for row in rows
                for cell in row
                if isinstance(cell, str) and any(ch.isdigit() for ch in cell)
            )

        # 1) Table region detection (per page)
        table_regions_per_page = self._detect_table_regions(doc)

        # 2) Pass 1: collect block-level text & font statistics
        font_size_counts: Dict[int, int] = {}
        block_infos: List[Dict[str, Any]] = []

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            page_dict = page.get_text("dict")

            regions = table_regions_per_page.get(page_index, [])

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # 0 = text
                    continue

                block_bbox = block.get("bbox", None)

                block_text_parts: List[str] = []
                max_font_size = 0.0
                any_bold = False
                # Determine whether this block lies inside a table region
                block_in_table = False

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "")
                        if not text or not text.strip():
                            continue
                        size = float(span.get("size", 0.0))
                        flags = int(span.get("flags", 0))
                        font_size = int(round(size))
                        font_size_counts[font_size] = font_size_counts.get(
                            font_size, 0
                        ) + len(text)

                        if size > max_font_size:
                            max_font_size = size
                        if flags & 1:
                            any_bold = True

                        # Table membership check (span centre vs detected regions)
                        if regions and not block_in_table:
                            span_bbox = span.get("bbox", None)
                            if span_bbox:
                                sx0, sy0, sx1, sy1 = span_bbox
                                cx = (sx0 + sx1) / 2.0
                                cy = (sy0 + sy1) / 2.0
                                for x0, y0, x1, y1 in regions:
                                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                                        block_in_table = True
                                        break

                        block_text_parts.append(text)

                block_text = " ".join(block_text_parts).strip()
                if not block_text:
                    continue

                block_infos.append(
                    {
                        "page_index": page_index,
                        "text": block_text,
                        "max_font_size": max_font_size,
                        "is_bold": any_bold,
                        "bbox": block_bbox,
                        "in_table": block_in_table,
                    }
                )

        # Determine base (body) font size
        if font_size_counts:
            base_font_size = max(font_size_counts, key=font_size_counts.get)
        else:
            base_font_size = 0

        # 3) Pass 2: build paragraphs & outline with refined heading heuristics
        text_chunks: List[Dict[str, Any]] = []
        outline: List[Tuple[int, str]] = []
        seen_headings: set = set()
        page_headings: Dict[int, str] = {}
        page_tag_context: Dict[int, Counter] = defaultdict(Counter)
        pseudo_table_blocks: Dict[int, List[str]] = defaultdict(list)

        for info in block_infos:
            text = info["text"].strip()
            size = info["max_font_size"]
            is_bold = bool(info["is_bold"])
            page_index = info.get("page_index", -1)
            if not text:
                continue

            if info["in_table"]:
                # Save for pseudo-table reconstruction if detector fails
                pseudo_table_blocks[page_index].append(text)
                continue

            # Heading size criterion
            is_heading_candidate = False
            if base_font_size > 0:
                if size > base_font_size * 1.2 or (
                    is_bold and size >= base_font_size * 1.1
                ):
                    is_heading_candidate = True

            # Additional filters to avoid false headings
            if is_heading_candidate:
                # Too long / likely a full sentence
                if len(text) > 120:
                    is_heading_candidate = False
                # Contains strong sentence punctuation (but allow colon)
                if any(p in text for p in [".", "?", "!"]):
                    is_heading_candidate = False
                # Bullet / list items
                stripped = text.lstrip()
                if stripped.startswith(("-", "•", "·", "*")):
                    is_heading_candidate = False
                # Purely numeric (page numbers, etc.)
                if stripped.replace(" ", "").isdigit():
                    is_heading_candidate = False
                # TOC-style line ending with page number (e.g. "About Bud APAC  2")
                tokens = stripped.split()
                if (
                    len(tokens) >= 2
                    and tokens[-1].isdigit()
                    and len(tokens[-1]) <= 3
                ):
                    is_heading_candidate = False

            if is_heading_candidate:
                if text not in seen_headings:
                    # Determine heading level by relative size
                    if size >= base_font_size * 2:
                        level = 1
                    elif size >= base_font_size * 1.5:
                        level = 2
                    else:
                        level = 3
                    outline.append((level, text))
                    seen_headings.add(text)
                    page_headings[page_index] = text
                # Do NOT add heading text to text_blocks
                continue

            # Non-heading -> paragraph block
            nearest_heading = page_headings.get(page_index)
            sub_chunks = self._split_block_text(text)
            for chunk_text in sub_chunks:
                chunk_payload = self._build_chunk_payload(
                    chunk_text,
                    page_index,
                    info.get("bbox"),
                    nearest_heading,
                )
                text_chunks.append(chunk_payload)
                if chunk_payload["tags"]:
                    weight = max(1, min(3, len(chunk_payload["text"]) // 250))
                    for tag in chunk_payload["tags"]:
                        page_tag_context[page_index][tag] += weight
            self._event_counters["text_chunks"] += len(sub_chunks)

        # 4) Table reconstruction using structure model (when available)
        table_payloads: List[Dict[str, Any]] = []
        if self.structure_model is not None and self.structure_processor is not None:
            for page_index, regions in table_regions_per_page.items():
                if not regions:
                    continue
                page = doc.load_page(page_index)
                pix = page.get_pixmap()
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )

                page_dict = page.get_text("dict")
                context_tags = None
                if page_index in page_tag_context:
                    context_tags = [
                        tag for tag, _ in page_tag_context[page_index].most_common(16)
                    ]

                for region in regions:
                    self._event_counters["tables_detected"] += 1
                    x0, y0, x1, y1 = region
                    # Slightly expand region to avoid clipping border text
                    margin = 2.0
                    rx0 = max(int(x0 - margin), 0)
                    ry0 = max(int(y0 - margin), 0)
                    rx1 = min(int(x1 + margin), pix.width)
                    ry1 = min(int(y1 + margin), pix.height)

                    table_image = img_array[ry0:ry1, rx0:rx1]

                    inputs = self.structure_processor(
                        images=table_image, return_tensors="pt"
                    )
                    outputs = self.structure_model(**inputs)

                    # Try to decode structure into row/column boxes
                    try:
                        structure = self.structure_processor.post_process_object_detection(
                            outputs,
                            threshold=0.5,
                            target_sizes=[(table_image.shape[0], table_image.shape[1])],
                        )
                    except Exception:
                        try:
                            structure = self.structure_processor.post_process(
                                outputs,
                                target_sizes=[(table_image.shape[0], table_image.shape[1])],
                            )
                        except Exception:
                            structure = []

                    row_boxes: List[List[float]] = []
                    col_boxes: List[List[float]] = []

                    if structure:
                        pred = structure[0]
                        labels = pred.get("labels", [])
                        boxes = pred.get("boxes", [])
                        for lbl, box in zip(labels, boxes):
                            try:
                                label_idx = int(lbl.item())
                            except Exception:
                                label_idx = int(lbl)
                            class_name = self.structure_model.config.id2label.get(
                                label_idx, ""
                            )
                            box_vals = box.tolist() if hasattr(box, "tolist") else box
                            if class_name == "table row":
                                row_boxes.append(box_vals)
                            elif class_name == "table column":
                                col_boxes.append(box_vals)

                        row_boxes.sort(key=lambda b: b[1])
                        col_boxes.sort(key=lambda b: b[0])

                    # Fallback: if structure is empty, treat each text line as a single-cell row
                    bbox_tuple = (float(rx0), float(ry0), float(rx1), float(ry1))

                    if not row_boxes or not col_boxes:
                        clip_rect = fitz.Rect(rx0, ry0, rx1, ry1)
                        fitz_rows = self._extract_table_with_fitz(page, clip_rect)
                        if fitz_rows:
                            processed_rows = self._expand_year_columns(fitz_rows)
                            table_payloads.append(
                                self._build_table_payload(
                                    processed_rows,
                                    page_index,
                                    bbox_tuple,
                                    page_level_tags=context_tags,
                                )
                            )
                            continue

                        raw_text = page.get_text("text", clip=clip_rect)
                        rows = [
                            [ln.strip()]
                            for ln in raw_text.splitlines()
                            if ln.strip()
                        ]
                        if rows:
                            processed_rows = self._expand_year_columns(rows)
                            table_payloads.append(
                                self._build_table_payload(
                                    processed_rows,
                                    page_index,
                                    bbox_tuple,
                                    page_level_tags=context_tags,
                                )
                            )
                        continue

                    # Build empty cell grid
                    n_rows = len(row_boxes)
                    n_cols = len(col_boxes)
                    table_cells: List[List[str]] = [
                        ["" for _ in range(n_cols)] for _ in range(n_rows)
                    ]

                    # Collect all text spans in the table region (page coordinate system)
                    table_spans: List[Tuple[float, float, float, float, str]] = []
                    for block in page_dict.get("blocks", []):
                        if block.get("type") != 0:
                            continue
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                span_text = span.get("text", "")
                                if not span_text.strip():
                                    continue
                                sb = span.get("bbox")
                                if not sb:
                                    continue
                                sx0, sy0, sx1, sy1 = sb
                                cx = (sx0 + sx1) / 2.0
                                cy = (sy0 + sy1) / 2.0
                                if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
                                    table_spans.append((cx, cy, sx0, sy0, span_text))

                    # Assign spans to cells based on centre position (relative to table region)
                    for cx, cy, sx0, sy0, span_text in table_spans:
                        rel_x = cx - rx0
                        rel_y = cy - ry0

                        row_idx: Optional[int] = None
                        col_idx: Optional[int] = None

                        for i, rb in enumerate(row_boxes):
                            ry0, ry1 = rb[1], rb[3]
                            if ry0 <= rel_y <= ry1:
                                row_idx = i
                                break

                        for j, cb in enumerate(col_boxes):
                            cx0, cx1 = cb[0], cb[2]
                            if cx0 <= rel_x <= cx1:
                                col_idx = j
                                break

                        if row_idx is None or col_idx is None:
                            continue

                        existing = table_cells[row_idx][col_idx]
                        table_cells[row_idx][col_idx] = (
                            f"{existing} {span_text.strip()}".strip()
                            if existing
                            else span_text.strip()
                        )

                    clip_rect = fitz.Rect(rx0, ry0, rx1, ry1)
                    filled_cells = _count_non_empty_cells(table_cells)
                    numeric_cells = _count_numeric_cells(table_cells)
                    total_cells = max(len(table_cells) * len(table_cells[0]), 1)
                    fill_ratio = filled_cells / total_cells

                    prefer_fitz = False
                    if filled_cells == 0 or numeric_cells < 3 or fill_ratio < 0.25 or len(table_cells[0]) <= 2:
                        fitz_rows = self._extract_table_with_fitz(page, clip_rect)
                        if fitz_rows:
                            fitz_numeric = _count_numeric_cells(fitz_rows)
                            fitz_filled = _count_non_empty_cells(fitz_rows)
                            if fitz_numeric > numeric_cells or fitz_filled > filled_cells:
                                processed_rows = self._expand_year_columns(fitz_rows)
                                table_payloads.append(
                                    self._build_table_payload(
                                        processed_rows,
                                        page_index,
                                        bbox_tuple,
                                        page_level_tags=context_tags,
                                    )
                                )
                                prefer_fitz = True

                    if prefer_fitz:
                        continue

                    if filled_cells == 0:
                        raw_text = page.get_text("text", clip=clip_rect)
                        rows = [
                            [ln.strip()]
                            for ln in raw_text.splitlines()
                            if ln.strip()
                        ]
                        if rows:
                            processed_rows = self._expand_year_columns(rows)
                            table_payloads.append(
                                self._build_table_payload(
                                    processed_rows,
                                    page_index,
                                    bbox_tuple,
                                    page_level_tags=context_tags,
                                )
                            )
                        continue

                    processed_rows = self._expand_year_columns(table_cells)
                    table_payloads.append(
                        self._build_table_payload(
                            processed_rows,
                            page_index,
                            bbox_tuple,
                            page_level_tags=context_tags,
                        )
                    )

        # Pseudo-table fallback: convert stored table blocks into lightweight tables
        for page_index, blocks in pseudo_table_blocks.items():
            has_detected_table = any(tbl.get("page") == page_index for tbl in table_payloads)
            if has_detected_table:
                continue
            pseudo_rows = self._blocks_to_pseudo_rows(blocks)
            if not pseudo_rows:
                continue
            context_tags = None
            if page_index in page_tag_context:
                context_tags = [
                    tag for tag, _ in page_tag_context[page_index].most_common(16)
                ]
            table_payloads.append(
                self._build_table_payload(
                    pseudo_rows,
                    page_index,
                    None,
                    page_level_tags=context_tags,
                    source_label="pseudo_table",
                )
            )
            self._event_counters["tables_pseudo"] += 1

        page_count = doc.page_count
        doc.close()
        self._processing_report = {
            "pages": page_count,
            "text_chunks": len(text_chunks),
            "tables_extracted": len(table_payloads),
            "tables_suspect": self._event_counters.get("tables_suspect", 0),
            "events": dict(self._event_counters),
            "page_tags": {
                page: page_tags.most_common(8)
                for page, page_tags in page_tag_context.items()
            },
        }
        return text_chunks, outline, table_payloads

    # -------------------------------------------------------------------------
    # Knowledge base construction
    # -------------------------------------------------------------------------

    def build_knowledge_base(
        self,
        text_chunks: List[Dict[str, Any]],
        outline: List[Tuple[int, str]],
        tables: List[Dict[str, Any]],
        source_pdf: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build multi-type knowledge base:
        - text.summaries / text.embeddings / text.original_texts
        - outline.headers / outline.levels / outline.embeddings
        - tables.phrases / tables.table_ids / tables.embeddings / tables.tables
        """
        kb: Dict[str, Any] = {"text": {}, "outline": {}, "tables": {}}

        # 1) Textual content: summarise then embed
        summaries: List[str] = []
        original_texts: List[str] = []
        provenance_records: List[Dict[str, Any]] = []
        summarizer_ready = (
            self.summarizer_tokenizer is not None and self.summarizer_model is not None
        )

        for chunk in text_chunks:
            chunk_text = chunk.get("text", "")
            original_texts.append(chunk_text)
            provenance_records.append(
                {
                    "page": chunk.get("page"),
                    "bbox": chunk.get("bbox"),
                    "heading": chunk.get("heading"),
                    "chunk_id": chunk.get("chunk_id"),
                    "hash": chunk.get("hash"),
                    "tags": chunk.get("tags", []),
                }
            )

            cleaned_text = chunk_text.strip()
            if not cleaned_text:
                summaries.append("")
                continue

            summary_text = ""
            if summarizer_ready:
                try:
                    inputs = self.summarizer_tokenizer.encode(
                        "summarize: " + cleaned_text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                    )
                    summary_ids = self.summarizer_model.generate(
                        inputs,
                        max_length=100,
                        num_beams=4,
                        early_stopping=True,
                    )
                    summary_text = self.summarizer_tokenizer.decode(
                        summary_ids[0], skip_special_tokens=True
                    )
                except Exception as exc:  # pragma: no cover - defensive path
                    summarizer_ready = False
                    self._event_counters["summaries_failure"] += 1
                    logger.warning("Summarizer failed, switching to fallback: %s", exc)

            if not summary_text:
                metadata_summary = self._metadata_fallback_summary(chunk.get("tags", []))
                if metadata_summary:
                    self._event_counters["summaries_metadata_fallback"] += 1
                    summary_text = metadata_summary
                else:
                    self._event_counters["summaries_fallback"] += 1
                    summary_text = cleaned_text
                    if len(summary_text) > 400:
                        summary_text = summary_text[:397] + "..."

            summaries.append(summary_text.strip())

        if self.embedding_model is not None:
            text_vectors = self.embedding_model.encode(
                summaries, show_progress_bar=False
            )
        else:
            text_vectors = np.empty((0, 0), dtype=np.float32)

        kb["text"]["summaries"] = summaries
        kb["text"]["original_texts"] = original_texts
        kb["text"]["metadata"] = provenance_records
        kb["text"]["embeddings"] = text_vectors

        # 2) Outline: embed headings
        headers = [h for (_lvl, h) in outline]
        levels = [lvl for (lvl, _h) in outline]
        if self.embedding_model is not None and headers:
            outline_vectors = self.embedding_model.encode(
                headers, show_progress_bar=False
            )
        else:
            outline_vectors = np.empty((0, 0), dtype=np.float32)

        kb["outline"]["headers"] = headers
        kb["outline"]["levels"] = levels
        kb["outline"]["embeddings"] = outline_vectors

        # 3) Tables: extract key indicator phrases (usually first column of data rows)
        table_phrases: List[str] = []
        table_ids: List[int] = []
        table_rows: List[List[List[str]]] = []
        table_metadata: List[Dict[str, Any]] = []
        for t_index, table_payload in enumerate(tables):
            table = table_payload.get("rows", [])
            if not table:
                continue

            # Heuristic: treat first row as header if it looks like one
            start_row = 0
            first_row = table[0]
            if first_row:
                first_cell = str(first_row[0]).strip().lower()
                generic_headers = {
                    "indicator",
                    "indicators",
                    "metric",
                    "metrics",
                    "parameter",
                    "parameters",
                    "year",
                    "years",
                    "category",
                    "categories",
                    "item",
                    "items",
                    "unit",
                    "units",
                }
                header_row = False
                if first_cell in generic_headers:
                    header_row = True
                else:
                    # if many of the other cells look numeric (years / figures), treat row as header
                    numeric_count = 0
                    non_empty = 0
                    for cell in first_row[1:]:
                        s = str(cell).strip()
                        if not s:
                            continue
                        non_empty += 1
                        if s.replace(".", "").replace(",", "").isdigit():
                            numeric_count += 1
                    if non_empty > 0 and numeric_count / non_empty > 0.5:
                        header_row = True
                if header_row:
                    start_row = 1

            indicator_found = False
            for row in table[start_row:]:
                if not row:
                    continue
                first_cell_text = str(row[0]).strip()
                if not first_cell_text:
                    continue
                # skip pure numeric keys
                if first_cell_text.replace(".", "").isdigit():
                    continue
                # at least one numeric cell in the remaining columns => likely an indicator row
                has_numeric = False
                for cell in row[1:]:
                    s = str(cell)
                    if any(ch.isdigit() for ch in s):
                        has_numeric = True
                        break
                if not has_numeric:
                    continue

                table_phrases.append(first_cell_text)
                table_ids.append(t_index)

                indicator_found = True

            if not indicator_found and table:
                fallback_phrase = str(table[0][0]).strip()
                if fallback_phrase:
                    table_phrases.append(fallback_phrase)
                    table_ids.append(t_index)

            table_rows.append(table)
            table_metadata.append(
                {
                    key: val
                    for key, val in table_payload.items()
                    if key != "rows"
                }
            )

        if self.embedding_model is not None and table_phrases:
            table_vectors = self.embedding_model.encode(
                table_phrases, show_progress_bar=False
            )
        else:
            table_vectors = np.empty((0, 0), dtype=np.float32)

        kb["tables"]["phrases"] = table_phrases
        kb["tables"]["table_ids"] = table_ids
        kb["tables"]["embeddings"] = table_vectors
        kb["tables"]["tables"] = table_rows
        kb["tables"]["metadata"] = table_metadata

        kb["metadata"] = self._build_kb_metadata(
            source_pdf=str(source_pdf) if source_pdf else None,
            text_chunks=text_chunks,
            tables=tables,
        )

        self._processing_report.setdefault("events", {})
        self._processing_report["events"].update(dict(self._event_counters))
        self._processing_report["text_chunks"] = len(text_chunks)
        self._processing_report["tables_extracted"] = len(table_rows)

        return kb

    # -------------------------------------------------------------------------
    # FAISS indexing
    # -------------------------------------------------------------------------

    def store_faiss_indexes(self, knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store text / outline / table embeddings into FAISS inner-product indices
        (with vectors L2-normalised to approximate cosine similarity).
        """
        indexes: Dict[str, Any] = {}

        def _norm(v: np.ndarray) -> np.ndarray:
            if v.size == 0:
                return v
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            return v / norms

        # Text
        text_vecs = np.array(
            knowledge_base["text"].get("embeddings", np.empty((0, 0))),
            dtype=np.float32,
        )
        text_vecs = _norm(text_vecs)
        if text_vecs.size > 0:
            index_text = faiss.IndexFlatIP(text_vecs.shape[1])
            index_text.add(text_vecs)
        else:
            index_text = None
        indexes["text_index"] = index_text

        # Outline
        outline_vecs = np.array(
            knowledge_base["outline"].get("embeddings", np.empty((0, 0))),
            dtype=np.float32,
        )
        outline_vecs = _norm(outline_vecs)
        if outline_vecs.size > 0:
            index_outline = faiss.IndexFlatIP(outline_vecs.shape[1])
            index_outline.add(outline_vecs)
        else:
            index_outline = None
        indexes["outline_index"] = index_outline

        # Tables
        table_vecs = np.array(
            knowledge_base["tables"].get("embeddings", np.empty((0, 0))),
            dtype=np.float32,
        )
        table_vecs = _norm(table_vecs)
        if table_vecs.size > 0:
            index_table = faiss.IndexFlatIP(table_vecs.shape[1])
            index_table.add(table_vecs)
        else:
            index_table = None
        indexes["table_index"] = index_table

        knowledge_base["text"]["index"] = index_text
        knowledge_base["outline"]["index"] = index_outline
        knowledge_base["tables"]["index"] = index_table
        return indexes


# -------------------------------------------------------------------------
# Helper functions for saving / CLI
# -------------------------------------------------------------------------


def _kb_to_json_payload(knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serialisable snapshot of the knowledge base (without FAISS indexes)."""

    def _array_to_list(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    text_section = knowledge_base.get("text", {})
    outline_section = knowledge_base.get("outline", {})
    table_section = knowledge_base.get("tables", {})

    payload: Dict[str, Any] = {
        "text": {
            "summaries": text_section.get("summaries", []),
            "original_texts": text_section.get("original_texts", []),
            "metadata": text_section.get("metadata", []),
            "embeddings": _array_to_list(
                text_section.get("embeddings", np.empty((0, 0)))
            ),
        },
        "outline": {
            "headers": outline_section.get("headers", []),
            "levels": outline_section.get("levels", []),
            "embeddings": _array_to_list(
                outline_section.get("embeddings", np.empty((0, 0)))
            ),
        },
        "tables": {
            "phrases": table_section.get("phrases", []),
            "table_ids": table_section.get("table_ids", []),
            "metadata": table_section.get("metadata", []),
            "embeddings": _array_to_list(
                table_section.get("embeddings", np.empty((0, 0)))
            ),
            "tables": table_section.get("tables", []),
        },
        "metadata": knowledge_base.get("metadata", {}),
    }
    return payload


def _save_processing_outputs(
    pdf_path: Path,
    knowledge_base: Dict[str, Any],
    indexes: Dict[str, Any],
    output_dir: Path,
    processing_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Save knowledge_base JSON and FAISS indexes to the given directory.
    Returns a mapping of names to saved paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = pdf_path.stem

    kb_json_path = output_dir / f"{base_name}_kb.json"
    with kb_json_path.open("w", encoding="utf-8") as f:
        json.dump(_kb_to_json_payload(knowledge_base), f, ensure_ascii=False, indent=2)

    saved_paths: Dict[str, Path] = {"knowledge_base": kb_json_path}

    # Save FAISS indexes if available
    if indexes.get("text_index") is not None:
        text_idx_path = output_dir / f"{base_name}_text.index"
        faiss.write_index(indexes["text_index"], str(text_idx_path))
        saved_paths["text_index"] = text_idx_path

    if indexes.get("outline_index") is not None:
        outline_idx_path = output_dir / f"{base_name}_outline.index"
        faiss.write_index(indexes["outline_index"], str(outline_idx_path))
        saved_paths["outline_index"] = outline_idx_path

    if indexes.get("table_index") is not None:
        table_idx_path = output_dir / f"{base_name}_table.index"
        faiss.write_index(indexes["table_index"], str(table_idx_path))
        saved_paths["table_index"] = table_idx_path

    if processing_report:
        report_path = output_dir / f"{base_name}_processing_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(processing_report, f, ensure_ascii=False, indent=2)
        saved_paths["processing_report"] = report_path

    return saved_paths


def _collect_pdf_paths(target_path: Path) -> List[Path]:
    """Collect all PDF files from a path (single file or directory)."""
    if target_path.is_file() and target_path.suffix.lower() == ".pdf":
        return [target_path]

    if target_path.is_dir():
        return sorted(
            p for p in target_path.rglob("*.pdf") if p.is_file()
        )

    return []


def _process_single_pdf(
    preprocessor: ESGReportPreprocessor, pdf_path: Path, output_dir: Path
) -> None:
    """Run the full preprocessing pipeline on a single PDF and save outputs."""
    text_chunks, outline, tables = preprocessor.parse_pdf(str(pdf_path))
    kb = preprocessor.build_knowledge_base(
        text_chunks, outline, tables, source_pdf=str(pdf_path)
    )
    indexes = preprocessor.store_faiss_indexes(kb)
    processing_report = preprocessor.get_processing_report()

    print(
        f"Parsed {len(text_chunks)} text chunks, "
        f"{len(outline)} outline entries, and "
        f"{len(tables)} tables from {pdf_path}."
    )

    if indexes["text_index"] is not None:
        print(f"  - Text index contains {indexes['text_index'].ntotal} vectors.")
    if indexes["outline_index"] is not None:
        print(f"  - Outline index contains {indexes['outline_index'].ntotal} vectors.")
    if indexes["table_index"] is not None:
        print(f"  - Table index contains {indexes['table_index'].ntotal} vectors.")

    saved = _save_processing_outputs(pdf_path, kb, indexes, output_dir, processing_report)
    print(f"  - Saved knowledge base to {saved['knowledge_base']}")
    for key in ("text_index", "outline_index", "table_index"):
        if key in saved:
            print(f"  - Saved {key.replace('_', ' ')} to {saved[key]}")
    if "processing_report" in saved:
        print(f"  - Saved processing report to {saved['processing_report']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process ESG PDF reports into FAISS-ready knowledge bases (ESGReveal-style preprocessing)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help=(
            "PDF file or directory containing PDFs. "
            "Defaults to the 'report' folder next to this script."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Directory where processed outputs will be saved. "
            "Defaults to 'processed_reports' next to this script."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional HuggingFace cache directory to reuse model downloads.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (local_files_only=True for transformers models).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_reports_dir = script_dir / "report"
    default_output_dir = script_dir / "processed_reports"

    target_path = Path(args.path).resolve() if args.path else default_reports_dir
    output_root = Path(args.output_dir).resolve() if args.output_dir else default_output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    pdf_paths = _collect_pdf_paths(target_path)
    if not pdf_paths:
        print(f"No PDF files found at {target_path}.")
        raise SystemExit(1)

    print(f"Found {len(pdf_paths)} PDF(s) to process under {target_path}.")
    metadata_module = ESGMetadataModule()
    preprocessor = ESGReportPreprocessor(
        cache_dir=args.cache_dir,
        offline_mode=args.offline,
        metadata_module=metadata_module,
    )
    for pdf in pdf_paths:
        print(f"\nProcessing {pdf} ...")
        _process_single_pdf(preprocessor, pdf, output_root)


if __name__ == "__main__":
    main()
