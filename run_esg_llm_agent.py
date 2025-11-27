from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import faiss

from esg_llm_agent import ESGRevealAgent
from esg_metadata import ESGMetadataModule, ESGMetadataRecord


def _resolve_kb_path(path_str: str) -> Path:
    path = Path(path_str).resolve()
    if path.exists():
        return path
    if path.suffix == ".json" and path.stem.endswith("_kb"):
        legacy_dir = path.parent / path.stem[:-3]
        candidate = legacy_dir / "knowledge_base.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"KB path not found: {path_str}")


def load_processed_kb_and_indexes(kb_path: Path) -> Tuple[Dict[str, Any], Dict[str, Optional[faiss.Index]]]:
    """
    Load knowledge base JSON and FAISS indexes produced by esg_report_processing.py.

    Expects files named:
      <base>_kb.json
      <base>_text.index
      <base>_outline.index
      <base>_table.index
    """
    if kb_path.is_dir():
        kb_json_path = kb_path / "knowledge_base.json"
    else:
        kb_json_path = kb_path

    if not kb_json_path.is_file():
        raise FileNotFoundError(f"KB JSON not found: {kb_json_path}")

    with kb_json_path.open("r", encoding="utf-8") as f:
        kb_json = json.load(f)

    index_dir = kb_json_path.parent
    if kb_path.is_dir() or (kb_json_path.name == "knowledge_base.json"):
        text_index_path = index_dir / "text.index"
        outline_index_path = index_dir / "outline.index"
        table_index_path = index_dir / "table.index"
    else:
        base_name = kb_json_path.stem
        if base_name.endswith("_kb"):
            base_name = base_name[:-3]
        text_index_path = index_dir / f"{base_name}_text.index"
        outline_index_path = index_dir / f"{base_name}_outline.index"
        table_index_path = index_dir / f"{base_name}_table.index"

    # Rebuild knowledge_base dict in the same structure used by ESGReportPreprocessor
    kb: Dict[str, Any] = {
        "text": {
            "summaries": kb_json["text"].get("summaries", []),
            "original_texts": kb_json["text"].get("original_texts", []),
            "embeddings": np.array(kb_json["text"].get("embeddings", []), dtype=np.float32),
        },
        "outline": {
            "headers": kb_json["outline"].get("headers", []),
            "levels": kb_json["outline"].get("levels", []),
            "embeddings": np.array(kb_json["outline"].get("embeddings", []), dtype=np.float32),
        },
        "tables": {
            "phrases": kb_json["tables"].get("phrases", []),
            "table_ids": kb_json["tables"].get("table_ids", []),
            "embeddings": np.array(kb_json["tables"].get("embeddings", []), dtype=np.float32),
            "tables": kb_json["tables"].get("tables", []),
        },
    }

    # Load FAISS indexes if they exist
    indexes: Dict[str, Optional[faiss.Index]] = {}
    if text_index_path.is_file():
        indexes["text_index"] = faiss.read_index(str(text_index_path))
    else:
        indexes["text_index"] = None

    if outline_index_path.is_file():
        indexes["outline_index"] = faiss.read_index(str(outline_index_path))
    else:
        indexes["outline_index"] = None

    if table_index_path.is_file():
        indexes["table_index"] = faiss.read_index(str(table_index_path))
    else:
        indexes["table_index"] = None

    return kb, indexes


def pick_single_indicator(
    meta: ESGMetadataModule,
    aspect: str,
    kpi_contains: str | None = None,
) -> ESGMetadataRecord:
    """
    Select a single ESGMetadataRecord under a given aspect.

    - aspect: e.g. "A1. Emissions"
    - kpi_contains: substring to filter KPI text (case-insensitive).
      If None, use the first record under that aspect.
    """
    candidates = meta.list_by_aspect(aspect)
    if not candidates:
        raise ValueError(f"No indicators found for aspect: {aspect}")

    if kpi_contains:
        kpi_sub = kpi_contains.lower()
        candidates = [r for r in candidates if kpi_sub in r.kpi.lower()]
        if not candidates:
            raise ValueError(
                f"No indicators under aspect '{aspect}' whose KPI contains '{kpi_contains}'."
            )

    # Take the first matched record
    return candidates[0]


def run_single_indicator(
    agent: ESGRevealAgent,
    aspect: str,
    kpi_contains: str | None = None,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Check compliance for ONE indicator (ESGReveal-style).

    Returns the full result dict from ESGRevealAgent.answer_indicator().
    """
    meta = agent.metadata_module
    record = pick_single_indicator(meta, aspect=aspect, kpi_contains=kpi_contains)

    print(f"\n[Single Indicator] Aspect: {record.aspect}")
    print(f"KPI: {record.kpi}")
    if kpi_contains:
        print(f"Matched by KPI substring: '{kpi_contains}'")

    result = agent.answer_indicator(
        record,
        temperature=temperature,
    )

    raw_answer = result["raw_answer"]
    print("\nRaw LLM answer:")
    print(raw_answer)

    # Optional: try to parse JSON to show Disclosure field
    try:
        parsed = json.loads(raw_answer)
        disclosure = parsed.get("Disclosure", None)
        print(f"\nParsed Disclosure field: {disclosure}")
    except Exception:
        print("\n[WARN] Could not parse LLM answer as JSON. Check the model output.")

    return result


def run_all_indicators(
    agent: ESGRevealAgent,
    aspect: str | None = None,
    temperature: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Check compliance for ALL indicators (optionally restricted to one aspect).

    Returns a list of result dicts from ESGRevealAgent.answer_indicator().
    """
    meta = agent.metadata_module
    if aspect:
        records = meta.list_by_aspect(aspect)
        print(f"\n[All Indicators] Running for aspect '{aspect}' with {len(records)} indicators.")
    else:
        records = list(meta._all_records())  # type: ignore[attr-defined]
        print(f"\n[All Indicators] Running for ALL aspects, total {len(records)} indicators.")

    results: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records, start=1):
        print("\n--------------------------------------------")
        print(f"[{idx}/{len(records)}] Aspect: {rec.aspect}")
        print(f"KPI: {rec.kpi}")

        result = agent.answer_indicator(
            rec,
            temperature=temperature,
        )
        results.append(result)

        raw_answer = result["raw_answer"]
        # Try to parse Disclosure to give a quick glance
        disclosure_str = None
        try:
            parsed = json.loads(raw_answer)
            disclosure_str = parsed.get("Disclosure")
        except Exception:
            disclosure_str = None

        if disclosure_str is not None:
            print(f"Disclosure: {disclosure_str}")
        else:
            print("Disclosure: [could not parse JSON, see raw answer above]")

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Launch ESGReveal-style LLM Agent on a processed ESG report "
            "(Budweiser or others) using esg_llm_agent.py."
        )
    )
    parser.add_argument(
        "--kb-json",
        required=True,
        help=(
            "Path to a processed knowledge base JSON (either legacy <base>_kb.json or "
            "the new <report>/knowledge_base.json directory)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["single", "all"],
        default="single",
        help="single = one indicator; all = all indicators (optionally an aspect).",
    )
    parser.add_argument(
        "--aspect",
        type=str,
        default=None,
        help="ESG aspect, e.g. 'A1. Emissions'. Required for mode=single; optional for mode=all.",
    )
    parser.add_argument(
        "--kpi-contains",
        type=str,
        default=None,
        help="Substring to select a specific KPI when mode=single (case-insensitive).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM sampling temperature (default 0.1 for deterministic outputs).",
    )
    parser.add_argument(
        "--no-corom",
        action="store_true",
        help="Disable CoROM re-ranking even if available.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model name to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save all results as JSON (useful for mode=all).",
    )

    args = parser.parse_args()

    kb_input_path = _resolve_kb_path(args.kb_json)
    kb, indexes = load_processed_kb_and_indexes(kb_input_path)

    # Construct the agent wired to this processed report
    agent = ESGRevealAgent(
        knowledge_base=kb,
        faiss_indexes=indexes,
        use_corom=not args.no_corom,
        llm_model_name=args.model,
    )

    if args.mode == "single":
        if not args.aspect:
            raise SystemExit("For mode=single you MUST provide --aspect (e.g. 'A1. Emissions').")
        result = run_single_indicator(
            agent,
            aspect=args.aspect,
            kpi_contains=args.kpi_contains,
            temperature=args.temperature,
        )
        results = [result]
    else:  # mode == "all"
        results = run_all_indicators(
            agent,
            aspect=args.aspect,
            temperature=args.temperature,
        )

    # Optionally save everything to JSON
    if args.output_json:
        payload = []
        for r in results:
            rec: ESGMetadataRecord = r["record"]
            payload.append(
                {
                    "aspect": rec.aspect,
                    "kpi": rec.kpi,
                    "topic": rec.topic,
                    "quantity": rec.quantity,
                    "search_terms": rec.search_terms,
                    "knowledge": rec.knowledge,
                    "raw_answer": r["raw_answer"],
                }
            )
        out_path = Path(args.output_json).resolve()
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved {len(payload)} result(s) to {out_path}")


if __name__ == "__main__":
    main()
