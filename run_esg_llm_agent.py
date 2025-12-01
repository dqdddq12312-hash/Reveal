from __future__ import annotations

import argparse
import csv
import hashlib
import json
from json import JSONDecodeError
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from esg_llm_agent import ESGLLMAgent
from esg_metadata import ESGMetadataModule, ESGMetadataRecord


DISCLOSURE_LABELS = ["Yes", "Partial", "No", "Not Available"]
DEFAULT_PROCESSED_ROOT = Path("processed_reports")


@dataclass  
class RunConfig:
    processed_root: Path
    report: str
    mode: str
    aspect: Optional[str]
    kpi_contains: Optional[str]
    indicator_id: Optional[str]
    temperature: float
    model: str
    disable_reranker: bool
    embedding_model: str
    reranker_model: str
    top_k_per_source: int
    rerank_top_n: int
    output_json: Optional[Path]
    ground_truth: Optional[Path]
    metrics_output: Optional[Path]
    metrics_format: str
    checkpoint_path: Optional[Path]
    resume_from: Optional[Path]
    max_workers: int
    rate_limit: Optional[float]
    max_retries: int
    retry_wait: float
    inspect_indicator: Optional[str]
    inspect_mismatches: bool
    pytest_mode: bool
    consensus_runs: int
    consensus_delta: float
    min_text_evidence: int
    min_table_evidence: int
    min_total_evidence: int
    debug_mode: bool


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def normalize_disclosure(value: Optional[str]) -> str:
    if not value:
        return "Not Available"
    lookup = {
        "yes": "Yes",
        "no": "No",
        "partial": "Partial",
        "not available": "Not Available",
        "na": "Not Available",
    }
    return lookup.get(value.strip().lower(), value)


def extract_disclosure_from_result(result: Dict[str, Any]) -> Tuple[str, str]:
    candidate: Optional[str] = None
    source = "unknown"
    consensus = result.get("consensus")
    if isinstance(consensus, dict):
        final_answer = consensus.get("final_answer")
        if isinstance(final_answer, dict):
            candidate = final_answer.get("Disclosure")
            source = "consensus"
    if candidate is None:
        parsed = result.get("parsed_answer")
        if isinstance(parsed, dict):
            candidate = parsed.get("Disclosure")
            source = "parsed_answer"
    if candidate is None:
        raw_answer = result.get("raw_answer")
        if isinstance(raw_answer, str):
            try:
                decoded = json.loads(raw_answer)
                candidate = decoded.get("Disclosure")
                source = "raw_answer"
            except JSONDecodeError:
                candidate = None
    return normalize_disclosure(candidate), source


def _indicator_id_from_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if not metadata:
        return None
    return metadata.get("indicator_id")


def compute_evidence_hashes(evidence: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    hashes: List[Dict[str, Any]] = []
    if not evidence:
        return hashes
    for ev in evidence:
        content = ev.get("content", "")
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
        hashes.append(
            {
                "source_type": ev.get("source_type"),
                "hash": digest,
                "score": ev.get("hybrid_score") or ev.get("rerank_score") or ev.get("score"),
            }
        )
    return hashes


def summarize_calibration(calibration_logs: Optional[List[Optional[Dict[str, Any]]]]) -> Dict[str, Any]:
    if not calibration_logs:
        return {"status": "missing", "issues": []}
    issues: List[str] = []
    downgraded = False
    for log in calibration_logs:
        if not log:
            continue
        if not log.get("numeric_match", True):
            issues.append("numeric_mismatch")
        if not log.get("unit_match", True):
            issues.append("unit_mismatch")
        if log.get("downgraded"):
            downgraded = True
    status = "ok" if not issues else ("downgraded" if downgraded else "warning")
    return {"status": status, "issues": sorted(set(issues)), "downgraded": downgraded}


def build_result_envelope(
    report: str,
    record: ESGMetadataRecord,
    agent_result: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = agent_result.get("metadata") or record.to_json()
    evidence = agent_result.get("evidence") or []
    calibration_logs = agent_result.get("calibration_logs")
    envelope: Dict[str, Any] = {
        "report": report,
        "indicator_id": metadata.get("indicator_id", record.indicator_id),
        "metadata": metadata,
        "answer": agent_result.get("parsed_answer"),
        "answers": agent_result.get("parsed_answers"),
        "raw_answers": agent_result.get("raw_answers")
        or ([agent_result["raw_answer"]] if agent_result.get("raw_answer") else []),
        "consensus": agent_result.get("consensus"),
        "validation_logs": agent_result.get("validation_logs"),
        "calibration_logs": calibration_logs,
        "confidence": (agent_result.get("consensus") or {}).get("agreement"),
        "evidence": evidence,
        "evidence_hashes": compute_evidence_hashes(evidence),
        "calibration_verdict": summarize_calibration(calibration_logs),
        "evidence_metrics": agent_result.get("evidence_metrics"),
        "timestamp_utc": _now_iso(),
        "status": "ok",
    }
    label, source = extract_disclosure_from_result(agent_result)
    envelope["predicted_disclosure"] = label
    envelope["prediction_source"] = source
    return envelope


class RateLimiter:
    def __init__(self, per_minute: float) -> None:
        self.interval = 60.0 / per_minute if per_minute else 0.0
        self.lock = threading.Lock()
        self._next_time = time.monotonic()

    def acquire(self) -> None:
        if self.interval <= 0:
            return
        while True:
            with self.lock:
                now = time.monotonic()
                wait = self._next_time - now
                if wait <= 0:
                    self._next_time = now + self.interval
                    return
            time.sleep(wait)


class ResultStore:
    def __init__(self, existing: Optional[List[Dict[str, Any]]] = None, checkpoint_path: Optional[Path] = None) -> None:
        self.results: List[Dict[str, Any]] = existing or []
        self.checkpoint_path = checkpoint_path
        self.index: Dict[str, Dict[str, Any]] = {}
        for item in self.results:
            indicator_id = item.get("indicator_id") or _indicator_id_from_metadata(item.get("metadata"))
            if indicator_id:
                self.index[indicator_id] = item

    def add(self, result: Dict[str, Any]) -> None:
        indicator_id = result.get("indicator_id")
        if indicator_id and indicator_id in self.index:
            for idx, existing in enumerate(self.results):
                existing_id = existing.get("indicator_id") or _indicator_id_from_metadata(existing.get("metadata"))
                if existing_id == indicator_id:
                    self.results[idx] = result
                    break
        else:
            self.results.append(result)
        if indicator_id:
            self.index[indicator_id] = result
        self.flush()

    def flush(self) -> None:
        if not self.checkpoint_path:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_results(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Resume file must contain a JSON list of results.")
    return data


def pick_single_indicator(
    meta: ESGMetadataModule,
    aspect: str,
    kpi_contains: Optional[str] = None,
) -> ESGMetadataRecord:
    candidates = meta.list_by_aspect(aspect)
    if not candidates:
        raise ValueError(f"No indicators found for aspect: {aspect}")
    if kpi_contains:
        needle = kpi_contains.lower()
        candidates = [rec for rec in candidates if needle in rec.kpi.lower()]
        if not candidates:
            raise ValueError(
                f"No indicators in aspect '{aspect}' matched substring '{kpi_contains}'."
            )
    return candidates[0]


def select_records(
    meta: ESGMetadataModule,
    mode: str,
    aspect: Optional[str],
    kpi_contains: Optional[str],
    indicator_id: Optional[str],
) -> List[ESGMetadataRecord]:
    def _all_public_records() -> List[ESGMetadataRecord]:
        records: List[ESGMetadataRecord] = []
        for aspect_name in meta.list_aspects():
            records.extend(meta.list_by_aspect(aspect_name))
        return records

    if indicator_id:
        match = next((rec for rec in _all_public_records() if rec.indicator_id == indicator_id), None)
        if not match:
            raise ValueError(f"Indicator ID not found: {indicator_id}")
        return [match]
    if mode == "single":
        if not aspect:
            raise ValueError("--aspect is required for mode=single unless --indicator-id is provided.")
        return [pick_single_indicator(meta, aspect, kpi_contains)]
    records: Iterable[ESGMetadataRecord]
    if aspect:
        records = meta.list_by_aspect(aspect)
    else:
        records = _all_public_records()
    if kpi_contains:
        needle = kpi_contains.lower()
        records = [rec for rec in records if needle in rec.kpi.lower()]
    return list(records)


def execute_with_retries(
    agent: ESGLLMAgent,
    report: str,
    record: ESGMetadataRecord,
    cfg: RunConfig,
    rate_limiter: Optional[RateLimiter],
) -> Dict[str, Any]:
    attempts = 0
    last_error: Optional[str] = None
    while attempts <= cfg.max_retries:
        attempts += 1
        try:
            if rate_limiter:
                rate_limiter.acquire()
            agent_result = agent.answer_indicator(report, record)
            payload = build_result_envelope(report, record, agent_result)
            payload["attempts"] = attempts
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            wait = cfg.retry_wait * attempts
            if cfg.pytest_mode:
                print(f"Retrying {record.indicator_id} after error: {last_error}")
            time.sleep(wait)
    failure = {
        "report": report,
        "indicator_id": record.indicator_id,
        "metadata": record.to_json(),
        "status": "error",
        "error": last_error,
        "attempts": attempts,
        "timestamp_utc": _now_iso(),
    }
    return failure


def run_records(
    agent: ESGLLMAgent,
    report: str,
    records: List[ESGMetadataRecord],
    cfg: RunConfig,
    store: ResultStore,
    rate_limiter: Optional[RateLimiter],
) -> None:
    pending = [rec for rec in records if rec.indicator_id not in store.index]
    if not pending:
        print("All requested indicators already present in resume data.")
        return
    total = len(pending)
    print(f"Processing {total} indicator(s) for report '{report}'...")

    def _process(rec: ESGMetadataRecord) -> Dict[str, Any]:
        return execute_with_retries(agent, report, rec, cfg, rate_limiter)

    if cfg.max_workers <= 1:
        for idx, record in enumerate(pending, start=1):
            result = _process(record)
            store.add(result)
            disclosure = result.get("predicted_disclosure") or result.get("status")
            print(f"[{idx}/{total}] {record.indicator_id} -> {disclosure}")
    else:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            futures = {pool.submit(_process, rec): rec for rec in pending}
            completed = 0
            for future in as_completed(futures):
                rec = futures[future]
                result = future.result()
                store.add(result)
                completed += 1
                disclosure = result.get("predicted_disclosure") or result.get("status")
                print(f"[{completed}/{total}] {rec.indicator_id} -> {disclosure}")


def load_ground_truth(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    truth: Dict[str, Dict[str, Any]] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                truth[key] = value
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("indicator_id"):
                truth[item["indicator_id"]] = item
    else:
        raise ValueError("Ground-truth file must be a dict or list of dicts.")
    return truth


def evaluate_predictions(results: List[Dict[str, Any]], truth_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in results:
        indicator_id = item.get("indicator_id") or _indicator_id_from_metadata(item.get("metadata"))
        if indicator_id:
            lookup[indicator_id] = item
    confusion = {expected: {pred: 0 for pred in DISCLOSURE_LABELS} for expected in DISCLOSURE_LABELS}
    per_indicator: List[Dict[str, Any]] = []
    covered_rows = 0
    match_count = 0
    missing: List[str] = []
    for indicator_id, truth in truth_map.items():
        expected = normalize_disclosure(
            truth.get("expected_disclosure")
            or truth.get("Disclosure")
            or truth.get("disclosure")
        )
        row = {
            "indicator_id": indicator_id,
            "expected": expected,
            "truth_value": truth.get("Value") or truth.get("value"),
        }
        prediction = lookup.get(indicator_id)
        if not prediction:
            row["predicted"] = None
            row["match"] = False
            row["confidence"] = None
            row["status"] = "missing"
            per_indicator.append(row)
            missing.append(indicator_id)
            continue
        predicted = prediction.get("predicted_disclosure")
        row["predicted"] = predicted
        row["confidence"] = prediction.get("confidence")
        row["status"] = prediction.get("status")
        match = predicted == expected
        row["match"] = match
        per_indicator.append(row)
        covered_rows += 1
        if match:
            match_count += 1
        if predicted in DISCLOSURE_LABELS and expected in DISCLOSURE_LABELS:
            confusion[expected][predicted] += 1
    unexpected = [rid for rid in lookup.keys() if rid not in truth_map]
    accuracy = match_count / covered_rows if covered_rows else 0.0

    def _metric(label: str) -> Tuple[float, float, float]:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in DISCLOSURE_LABELS if other != label)
        fn = sum(confusion[label][other] for other in DISCLOSURE_LABELS if other != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    metrics_by_label = {
        label: {
            "precision": round(_metric(label)[0], 4),
            "recall": round(_metric(label)[1], 4),
            "f1": round(_metric(label)[2], 4),
        }
        for label in DISCLOSURE_LABELS
    }
    macro_precision = sum(m["precision"] for m in metrics_by_label.values()) / len(DISCLOSURE_LABELS)
    macro_recall = sum(m["recall"] for m in metrics_by_label.values()) / len(DISCLOSURE_LABELS)
    macro_f1 = sum(m["f1"] for m in metrics_by_label.values()) / len(DISCLOSURE_LABELS)

    summary = {
        "truth_total": len(truth_map),
        "coverage": covered_rows,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
    }
    return {
        "per_indicator": per_indicator,
        "confusion": confusion,
        "metrics_by_label": metrics_by_label,
        "summary": summary,
        "missing_ids": missing,
        "unexpected_ids": unexpected,
    }


def render_markdown_report(evaluation: Dict[str, Any]) -> str:
    summary = evaluation["summary"]
    lines = ["### ESG Evaluation Summary", "", "| Metric | Value |", "| --- | --- |"]
    for key, label in [
        ("truth_total", "Ground-truth KPIs"),
        ("coverage", "Predicted KPIs"),
        ("missing", "Missing KPIs"),
        ("unexpected", "Predicted w/o Truth"),
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
    ]:
        lines.append(f"| {label} | {summary[key]} |")
    lines.append("")
    lines.append("#### Confusion Matrix")
    header = "| Expected \\ Pred | " + " | ".join(DISCLOSURE_LABELS) + " |"
    lines.extend([header, "|" + " --- |" * (len(DISCLOSURE_LABELS) + 1)])
    for expected in DISCLOSURE_LABELS:
        row = [expected] + [str(evaluation["confusion"][expected][pred]) for pred in DISCLOSURE_LABELS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_metrics_report(evaluation: Dict[str, Any], path: Path, fmt: str) -> None:
    fmt_lower = fmt if fmt != "auto" else path.suffix.lstrip(".").lower()
    fmt_lower = fmt_lower or "md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt_lower == "md":
        path.write_text(render_markdown_report(evaluation), encoding="utf-8")
    elif fmt_lower == "json":
        path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt_lower == "csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["indicator_id", "expected", "predicted", "match", "confidence", "truth_value"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in evaluation["per_indicator"]:
                writer.writerow({key: row.get(key) for key in fieldnames})
    else:
        raise ValueError(f"Unsupported metrics format: {fmt_lower}")


def write_results(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} result(s) to {path}")


def inspect_indicator(
    indicator_id: str,
    results_lookup: Dict[str, Dict[str, Any]],
    truth_map: Optional[Dict[str, Dict[str, Any]]],
) -> None:
    result = results_lookup.get(indicator_id)
    if not result:
        print(f"No run data found for indicator {indicator_id}.")
        return
    metadata = result.get("metadata", {})
    print("\n=== Indicator Insight ===")
    print(f"Indicator ID: {indicator_id}")
    print(f"Aspect: {metadata.get('aspect')}")
    print(f"KPI: {metadata.get('kpi')}")
    print(f"Predicted disclosure: {result.get('predicted_disclosure')}")
    print(f"Confidence: {result.get('confidence')}")
    print("\nParsed answer:")
    print(json.dumps(result.get("answer"), ensure_ascii=False, indent=2))
    if truth_map and indicator_id in truth_map:
        print("\nGround truth snippet:")
        print(json.dumps(truth_map[indicator_id], ensure_ascii=False, indent=2))
    evidence = result.get("evidence") or []
    if evidence:
        print("\nEvidence excerpts:")
        for idx, ev in enumerate(evidence[:2], start=1):
            content = (ev.get("content") or "").strip().replace("\n", " ")
            score = ev.get("score") or 0.0
            print(f"[{idx}] ({ev.get('source_type')}) score={score:.4f} -> {content[:400]}")
    print("==========================\n")


def build_agent(meta: ESGMetadataModule, cfg: RunConfig) -> ESGLLMAgent:
    agent = ESGLLMAgent(
        kb_root=cfg.processed_root,
        model_name=cfg.model,
        temperature=cfg.temperature,
        top_k_per_source=cfg.top_k_per_source,
        rerank_top_n=cfg.rerank_top_n,
        embedding_model_name=cfg.embedding_model,
        reranker_model_name=cfg.reranker_model,
        metadata_module=meta,
        min_text_evidence=cfg.min_text_evidence,
        min_table_evidence=cfg.min_table_evidence,
        min_total_evidence=cfg.min_total_evidence,
        consensus_runs=cfg.consensus_runs,
        consensus_temperature_delta=cfg.consensus_delta,
        debug_mode=cfg.debug_mode,
    )
    if cfg.disable_reranker:
        agent.reranker = None
    return agent


def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, RunConfig]:
    parser = argparse.ArgumentParser(description="Reliability runner for ESGLLMAgent")
    parser.add_argument("--report", required=True, help="Processed report directory under processed_root.")
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--mode", choices=["single", "all"], default="single")
    parser.add_argument("--aspect", type=str, default=None)
    parser.add_argument("--kpi-contains", type=str, default=None)
    parser.add_argument("--indicator-id", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--embedding-model", type=str, default="moka-ai/m3e-base")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--no-corom", action="store_true")
    parser.add_argument("--top-k-per-source", type=int, default=4)
    parser.add_argument("--rerank-top-n", type=int, default=12)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--ground-truth", type=str, default=None)
    parser.add_argument("--metrics-output", type=str, default=None)
    parser.add_argument("--metrics-format", choices=["auto", "md", "csv", "json"], default="auto")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--rate-limit", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    parser.add_argument("--inspect-indicator", type=str, default=None)
    parser.add_argument("--inspect-mismatches", action="store_true")
    parser.add_argument("--pytest-mode", action="store_true")
    parser.add_argument("--consensus-runs", type=int, default=1)
    parser.add_argument("--consensus-delta", type=float, default=0.15)
    parser.add_argument("--min-text-evidence", type=int, default=1)
    parser.add_argument("--min-table-evidence", type=int, default=0)
    parser.add_argument("--min-total-evidence", type=int, default=2)
    parser.add_argument("--debug-evidence", action="store_true")
    args = parser.parse_args(argv)

    cfg = RunConfig(
        processed_root=Path(args.processed_root).resolve(),
        report=args.report,
        mode=args.mode,
        aspect=args.aspect,
        kpi_contains=args.kpi_contains,
        indicator_id=args.indicator_id,
        temperature=args.temperature,
        model=args.model,
        disable_reranker=args.no_corom,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        top_k_per_source=max(1, args.top_k_per_source),
        rerank_top_n=max(1, args.rerank_top_n),
        output_json=Path(args.output_json).resolve() if args.output_json else None,
        ground_truth=Path(args.ground_truth).resolve() if args.ground_truth else None,
        metrics_output=Path(args.metrics_output).resolve() if args.metrics_output else None,
        metrics_format=args.metrics_format,
        checkpoint_path=Path(args.checkpoint).resolve() if args.checkpoint else None,
        resume_from=Path(args.resume_from).resolve() if args.resume_from else None,
        max_workers=max(1, args.max_workers),
        rate_limit=args.rate_limit,
        max_retries=max(0, args.max_retries),
        retry_wait=max(1.0, args.retry_wait),
        inspect_indicator=args.inspect_indicator,
        inspect_mismatches=args.inspect_mismatches,
        pytest_mode=args.pytest_mode,
        consensus_runs=max(1, args.consensus_runs),
        consensus_delta=args.consensus_delta,
        min_text_evidence=max(0, args.min_text_evidence),
        min_table_evidence=max(0, args.min_table_evidence),
        min_total_evidence=max(1, args.min_total_evidence),
        debug_mode=args.debug_evidence,
    )
    return args, cfg


def cli_main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args, cfg = parse_args(argv)
    metadata = ESGMetadataModule()
    records = select_records(metadata, cfg.mode, cfg.aspect, cfg.kpi_contains, cfg.indicator_id)
    if not records:
        raise SystemExit("No ESG metadata records matched the provided filters.")

    agent = build_agent(metadata, cfg)
    rate_limiter = RateLimiter(cfg.rate_limit) if cfg.rate_limit else None

    existing_results = load_results(cfg.resume_from)
    store = ResultStore(existing_results, cfg.checkpoint_path)

    run_records(agent, cfg.report, records, cfg, store, rate_limiter)

    truth_map = load_ground_truth(cfg.ground_truth) if cfg.ground_truth else None
    evaluation = None
    if truth_map:
        evaluation = evaluate_predictions(store.results, truth_map)
        print("\n" + render_markdown_report(evaluation) + "\n")
        if cfg.metrics_output:
            write_metrics_report(evaluation, cfg.metrics_output, cfg.metrics_format)
        if cfg.inspect_mismatches:
            mismatched = [row["indicator_id"] for row in evaluation["per_indicator"] if not row["match"]]
            for indicator_id in mismatched:
                inspect_indicator(indicator_id, store.index, truth_map)
    if cfg.inspect_indicator:
        inspect_indicator(cfg.inspect_indicator, store.index, truth_map)

    if cfg.output_json:
        write_results(store.results, cfg.output_json)

    return {"results": store.results, "evaluation": evaluation}


def run_cli(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Pytest-friendly entry point."""
    return cli_main(argv)


def main() -> None:
    cli_main()


if __name__ == "__main__":
    main()
