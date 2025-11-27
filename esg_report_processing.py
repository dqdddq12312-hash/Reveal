from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Any, Optional

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


class ESGReportPreprocessor:
    """
    Report preprocessing module following ESGReveal:
    - parse_pdf: layout & table extraction (paragraphs, outline, tables)
    - build_knowledge_base: mT5 summaries + m3e embeddings for text/outline/tables
    - store_faiss_indexes: build FAISS indices for fast retrieval
    """

    def __init__(self) -> None:
        # Table detection & structure recognition (Table-Transformer family)
        self.table_detection_model_name = "microsoft/table-transformer-detection"
        self.table_structure_model_name = "microsoft/table-transformer-structure-recognition"

        try:
            self.detection_processor = AutoProcessor.from_pretrained(
                self.table_detection_model_name
            )
            self.detection_model = AutoModelForObjectDetection.from_pretrained(
                self.table_detection_model_name
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
                self.table_structure_model_name
            )
            self.structure_model = AutoModelForObjectDetection.from_pretrained(
                self.table_structure_model_name
            )
        except Exception as e:
            print(
                f"[WARN] Could not load table structure model '{self.table_structure_model_name}': {e}"
            )
            self.structure_processor = None
            self.structure_model = None

        # Text summarisation (mT5-base)
        try:
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/mt5-base"
            )
        except Exception as e:
            print(f"[WARN] Could not load summariser (mt5-base): {e}")
            self.summarizer_tokenizer = None
            self.summarizer_model = None

        # Text embedding (m3e-base)
        try:
            self.embedding_model = SentenceTransformer("moka-ai/m3e-base")
        except Exception as e:
            print(f"[WARN] Could not load embedding model (m3e-base): {e}")
            self.embedding_model = None

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

    def parse_pdf(
        self, pdf_path: str
    ) -> Tuple[List[str], List[Tuple[int, str]], List[List[List[str]]]]:
        """
        Parse the PDF and perform layout extraction.
        Returns:
            text_blocks: list of paragraph-like text blocks (strings)
            outline: list of (level, heading_text) entries
            tables: list of tables, each table is a list of rows, each row a list of cell texts
        """
        doc = fitz.open(pdf_path)

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
        text_blocks: List[str] = []
        outline: List[Tuple[int, str]] = []
        seen_headings: set = set()

        for info in block_infos:
            if info["in_table"]:
                # Table text will be re-extracted when reconstructing tables
                continue

            text = info["text"].strip()
            size = info["max_font_size"]
            is_bold = bool(info["is_bold"])

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
                # Do NOT add heading text to text_blocks
                continue

            # Non-heading -> paragraph block
            text_blocks.append(text)

        # 4) Table reconstruction using structure model (when available)
        assembled_tables: List[List[List[str]]] = []
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

                for region in regions:
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
                    if not row_boxes or not col_boxes:
                        clip_rect = fitz.Rect(x0, y0, x1, y1)
                        raw_text = page.get_text("text", clip=clip_rect)
                        rows = [
                            [ln.strip()]
                            for ln in raw_text.splitlines()
                            if ln.strip()
                        ]
                        if rows:
                            assembled_tables.append(rows)
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
                                if x0 <= cx <= x1 and y0 <= cy <= y1:
                                    table_spans.append((cx, cy, sx0, sy0, span_text))

                    # Assign spans to cells based on centre position (relative to table region)
                    for cx, cy, sx0, sy0, span_text in table_spans:
                        rel_x = cx - x0
                        rel_y = cy - y0

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

                    assembled_tables.append(table_cells)

        doc.close()
        return text_blocks, outline, assembled_tables

    # -------------------------------------------------------------------------
    # Knowledge base construction
    # -------------------------------------------------------------------------

    def build_knowledge_base(
        self,
        text_blocks: List[str],
        outline: List[Tuple[int, str]],
        tables: List[List[List[str]]],
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
        summarizer_ready = (
            self.summarizer_tokenizer is not None and self.summarizer_model is not None
        )
        for text in text_blocks:
            if not text.strip():
                summaries.append("")
                continue
            if summarizer_ready:
                inputs = self.summarizer_tokenizer.encode(
                    "summarize: " + text,
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
            else:
                # Fallback: trimmed original text
                summary_text = text.strip()
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
        kb["text"]["original_texts"] = text_blocks
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
        for t_index, table in enumerate(tables):
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

        if self.embedding_model is not None and table_phrases:
            table_vectors = self.embedding_model.encode(
                table_phrases, show_progress_bar=False
            )
        else:
            table_vectors = np.empty((0, 0), dtype=np.float32)

        kb["tables"]["phrases"] = table_phrases
        kb["tables"]["table_ids"] = table_ids
        kb["tables"]["embeddings"] = table_vectors
        kb["tables"]["tables"] = tables

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
            "embeddings": _array_to_list(
                table_section.get("embeddings", np.empty((0, 0)))
            ),
            "tables": table_section.get("tables", []),
        },
    }
    return payload


def _save_processing_outputs(
    pdf_path: Path,
    knowledge_base: Dict[str, Any],
    indexes: Dict[str, Any],
    output_dir: Path,
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
    text_blocks, outline, tables = preprocessor.parse_pdf(str(pdf_path))
    kb = preprocessor.build_knowledge_base(text_blocks, outline, tables)
    indexes = preprocessor.store_faiss_indexes(kb)

    print(
        f"Parsed {len(text_blocks)} text blocks, "
        f"{len(outline)} outline entries, and "
        f"{len(tables)} tables from {pdf_path}."
    )

    if indexes["text_index"] is not None:
        print(f"  - Text index contains {indexes['text_index'].ntotal} vectors.")
    if indexes["outline_index"] is not None:
        print(f"  - Outline index contains {indexes['outline_index'].ntotal} vectors.")
    if indexes["table_index"] is not None:
        print(f"  - Table index contains {indexes['table_index'].ntotal} vectors.")

    saved = _save_processing_outputs(pdf_path, kb, indexes, output_dir)
    print(f"  - Saved knowledge base to {saved['knowledge_base']}")
    for key in ("text_index", "outline_index", "table_index"):
        if key in saved:
            print(f"  - Saved {key.replace('_', ' ')} to {saved[key]}")


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
    preprocessor = ESGReportPreprocessor()
    for pdf in pdf_paths:
        print(f"\nProcessing {pdf} ...")
        _process_single_pdf(preprocessor, pdf, output_root)


if __name__ == "__main__":
    main()
