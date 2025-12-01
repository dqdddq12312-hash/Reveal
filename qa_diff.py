#!/usr/bin/env python
"""Quick QA diff helper for ESG compliance outputs.

Usage example:
    python qa_diff.py --baseline assistant_compliance_review.json \
        --candidate budweiser_all_ESG_results.json

The script normalises disclosure labels and surfaces:
    * total indicator coverage
    * aggregate status distributions
    * mismatch counts grouped by (baseline, candidate)
    * sample mismatches per combo
Use this after each agent run to spot regressions such as
Disclosed→Not Available spikes caused by retrieval or guardrail issues.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

NORMALISE_BASELINE = {
    "disclosed": "Yes",
    "partial": "Partial",
    "missing": "Not Available",
}

NORMALISE_CANDIDATE = {
    "yes": "Yes",
    "partial": "Partial",
    "no": "No",
    "not available": "Not Available",
}


def load_baseline(path: Path) -> Dict[str, Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["indicator_id"]: row for row in payload}


def load_candidate(path: Path) -> Dict[str, Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["indicator_id"]: row for row in payload}


def normalise_baseline(status: str) -> str:
    return NORMALISE_BASELINE.get((status or "").strip().lower(), "Not Available")


def normalise_candidate(status: str | None) -> str:
    return NORMALISE_CANDIDATE.get((status or "").strip().lower(), "Not Available")


def extract_candidate_status(row: Dict) -> str:
    answer = row.get("answer") or {}
    disclosure = answer.get("Disclosure") or row.get("predicted_disclosure")
    return normalise_candidate(disclosure)


def compare(baseline: Dict[str, Dict], candidate: Dict[str, Dict]) -> Dict[str, Any]:
    combos = Counter()
    samples: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    common_ids = set(baseline).intersection(candidate)
    missing_in_candidate = sorted(set(baseline) - set(candidate))
    missing_in_baseline = sorted(set(candidate) - set(baseline))
    mismatches: List[Dict[str, Any]] = []
    guardrail_buckets = Counter()

    for indicator_id in sorted(common_ids):
        base_row = baseline[indicator_id]
        cand_row = candidate[indicator_id]
        b_status = base_row.get("assistant_disclosure", "Missing")
        c_status = extract_candidate_status(cand_row)
        normalized_base = normalise_baseline(b_status)
        combos[(normalized_base, c_status)] += 1
        sample_key = (normalized_base, c_status)
        if len(samples[sample_key]) < 5:
            samples[sample_key].append(
                f"{indicator_id}: {base_row['kpi'][:120]}"
            )
        if normalized_base != c_status:
            metrics = cand_row.get("evidence_metrics") or {}
            reason_code = metrics.get("reason_code")
            if reason_code:
                guardrail_buckets[reason_code] += 1
            mismatches.append(
                {
                    "indicator_id": indicator_id,
                    "baseline": normalized_base,
                    "candidate": c_status,
                    "kpi": base_row.get("kpi"),
                    "candidate_confidence": cand_row.get("confidence"),
                    "guardrail_reason": metrics.get("reason"),
                    "guardrail_reason_code": reason_code,
                    "override": metrics.get("override"),
                }
            )

    return {
        "total_indicators": len(baseline),
        "candidate_indicators": len(candidate),
        "overlap": len(common_ids),
        "missing_in_candidate": missing_in_candidate,
        "missing_in_baseline": missing_in_baseline,
        "combo_counts": combos,
        "combo_samples": samples,
        "mismatches": mismatches,
        "guardrail_reasons": guardrail_buckets,
    }


def format_summary(results: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Total indicators (baseline): {results['total_indicators']}")
    lines.append(f"Indicators in candidate: {results['candidate_indicators']}")
    lines.append(f"Overlap: {results['overlap']}")
    if results["missing_in_candidate"]:
        lines.append(
            "Missing in candidate: " + ", ".join(results["missing_in_candidate"][:10])
        )
    if results["missing_in_baseline"]:
        lines.append(
            "Missing in baseline: " + ", ".join(results["missing_in_baseline"][:10])
        )
    lines.append("\nMismatch breakdown (baseline → candidate):")
    for (base_status, cand_status), count in results["combo_counts"].most_common():
        lines.append(f"  {base_status} → {cand_status}: {count}")
        samples = results["combo_samples"][ (base_status, cand_status) ]
        if samples:
            lines.append("    e.g. " + "; ".join(samples))
    if results.get("guardrail_reasons"):
        lines.append("\nGuardrail reasons among mismatches:")
        for reason, count in results["guardrail_reasons"].most_common():
            lines.append(f"  {reason}: {count}")
    if results.get("mismatches"):
        lines.append("\nTop mismatched indicators:")
        for record in results["mismatches"][:5]:
            lines.append(
                f"  {record['indicator_id']}: {record['baseline']}→{record['candidate']} (reason={record.get('guardrail_reason_code') or 'n/a'})"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff ESG compliance outputs")
    parser.add_argument("--baseline", required=True, type=Path, help="assistant_compliance_review.json path")
    parser.add_argument("--candidate", required=True, type=Path, help="New agent output JSON path")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path to save raw diff stats as JSON")
    parser.add_argument("--details-json", type=Path, default=None, help="Optional path to write per-indicator mismatches as JSON")
    parser.add_argument("--details-csv", type=Path, default=None, help="Optional path to write per-indicator mismatches as CSV")
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    candidate = load_candidate(args.candidate)
    results = compare(baseline, candidate)
    if args.json_output:
        args.json_output.write_text(
            json.dumps(
                {
                    **results,
                    "combo_counts": {f"{k[0]} → {k[1]}": v for k, v in results["combo_counts"].items()},
                    "combo_samples": {f"{k[0]} → {k[1]}": v for k, v in results["combo_samples"].items()},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    if args.details_json:
        args.details_json.write_text(
            json.dumps(results["mismatches"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.details_csv and results["mismatches"]:
        with args.details_csv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "indicator_id",
                "baseline",
                "candidate",
                "kpi",
                "candidate_confidence",
                "guardrail_reason",
                "guardrail_reason_code",
                "override",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in results["mismatches"]:
                writer.writerow(row)
    print(format_summary(results))


if __name__ == "__main__":
    main()
