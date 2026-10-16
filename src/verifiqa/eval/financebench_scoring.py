"""Evaluation-only FinanceBench reference review shared by CLI and notebooks.

The stored candidate and unit are immutable scoring inputs. Reviewed references
apply to every model and decision; they never supply verifier formula authority.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from .candidate_scoring import candidate_matches_gold, claim_quantity, numeric_claim_comparison

DEFAULT_SCORING_VIEW = "source_reviewed"
SCORING_VIEWS = ("source_reviewed", "benchmark")


def _sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()


def _repo_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("FinanceBench evaluation paths must remain inside the repository")
    return path


def _index(rows, key, label):
    result = {}
    for row in rows:
        rid = row[key]
        if rid in result:
            raise ValueError(f"Duplicate {label} ID: {rid}")
        result[rid] = row
    return result


def _comparison_row(row, reference_type):
    """Interpret a literal point-change display without altering its saved quantity."""
    claim = claim_quantity(row)
    if (reference_type.get("unit") != "percentage points" or claim["unit"] != "percent"
            or not claim["valid"]):
        return row, None
    answer = str(row.get("claim_input_answer") or row.get("answer") or "").replace("\N{MINUS SIGN}", "-")
    number = re.compile(r"(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
    matches = list(number.finditer(answer))
    point_matches = []
    for match in matches:
        value = float(match[0].replace(",", ""))
        if not math.isclose(value, claim["value"], rel_tol=1e-12, abs_tol=1e-12):
            continue
        prefix, suffix = answer[:match.start()], answer[match.end():]
        points = re.match(r"\s*percentage[-\s]+points?\b", suffix, re.I)
        if (re.search(r"(?:[$€£]|\b(?:USD|EUR|GBP|TWD|dollars?|euros?))\s*$", prefix, re.I)
                or (points and re.match(r"\s*(?:[$€£]|USD\b|EUR\b|ratio\b|percent\b)", suffix[points.end():], re.I))
                or (not points and re.match(r"\s*(?:%|percent\b|ratio\b|days?\b|years?\b|USD\b|EUR\b|count\b)", suffix, re.I))):
            return row, None
        clause = re.split(r"[;:\n]", prefix)[-1]
        if points and not re.search(r"\b(?:inputs?|operands?|assumptions?|unrelated|examples?)\b", clause, re.I):
            point_matches.append(match)
    # A matching operand cannot lend its unit to a different final quantity.
    final = matches[-1].start() if matches else None
    outside = re.sub(r"\([^()]*\)", lambda m: m[0] if re.search(
        r"\b(?:final|answer|result|instead)\b", m[0], re.I) else " " * len(m[0]), answer)
    outside_matches = list(number.finditer(outside))
    outside_final = outside_matches[-1].start() if outside_matches else None
    literal = next((match for match in reversed(point_matches)
                    if match.start() in {final, outside_final}), None)
    if literal is None:
        return row, None
    comparison = {**row, "claimed_value_unit": "percentage points"}
    return comparison, {"kind": "literal_percentage_point_display", "stored_unit": claim["unit"],
        "comparison_unit": "percentage points", "value_preserved": True,
        "source_literal": answer[literal.start():literal.end()] + re.match(
            r"\s*percentage[-\s]+points?\b", answer[literal.end():], re.I)[0]}


class FinanceBenchScorer:
    @classmethod
    def from_root(cls, root):
        root = Path(root).resolve()
        path = _repo_path(root, "data/financebench_scoring_review.json")
        raw = path.read_bytes()
        review = json.loads(raw)
        dataset_path = _repo_path(root, review["dataset"]["path"])
        dataset_bytes = dataset_path.read_bytes()
        if _sha(dataset_bytes) != review["dataset"]["sha256"]:
            raise ValueError("FinanceBench dataset hash mismatch")
        dataset = _index([json.loads(line) for line in dataset_bytes.splitlines() if line.strip()],
                         "financebench_id", "dataset")
        if len(dataset) != review["dataset"]["questions"]:
            raise ValueError("FinanceBench dataset count mismatch")
        source_lines = {}
        def validate_identity(rid, record):
            canonical = dataset.get(rid)
            if (canonical is None or record["question"] != canonical["question"]
                    or str(record["raw_answer"]) != str(canonical["answer"])
                    or record["question_sha256"] != _sha(record["question"])
                    or record["raw_answer_sha256"] != _sha(record["raw_answer"])):
                raise ValueError(f"FinanceBench reviewed identity mismatch: {rid}")
            if not record.get("sources"):
                raise ValueError(f"FinanceBench reviewed sources missing: {rid}")
            for source in record.get("sources", []):
                source_path = _repo_path(root, source["path"])
                if source_path not in source_lines:
                    source_lines[source_path] = source_path.read_text().splitlines()
                source_row = json.loads(source_lines[source_path][source["line"] - 1])
                if source_row.get("financebench_id", source_row.get("id")) != source["record_id"]:
                    raise ValueError(f"FinanceBench source identity mismatch: {rid}")
                value = source_row
                for part in re.sub(r"\[(\d+)\]", r".\1", source["field"]).split("."):
                    value = value[int(part)] if isinstance(value, list) else value[part]
                if (_sha(value) != source["field_sha256"]
                        or not isinstance(value, str) or source["quote"] not in value):
                    raise ValueError(f"FinanceBench source field/quote mismatch: {rid}")
        records = _index(review["records"], "id", "reference")
        for rid, record in records.items():
            validate_identity(rid, record)
            target_key = "plausible_targets" if record.get("kind") == "ambiguous_reference_scope" else "targets"
            if not record.get(target_key):
                raise ValueError(f"FinanceBench reviewed targets missing: {rid}")
            for target in record[target_key]:
                if (isinstance(target.get("value"), bool)
                        or not isinstance(target.get("value"), (int, float))
                        or not math.isfinite(target["value"]) or not target.get("unit")):
                    raise ValueError(f"Invalid FinanceBench reviewed quantity: {rid}")
        reference_units = review.get("reference_units", {})
        if set(reference_units) != set(dataset):
            raise ValueError("FinanceBench reference types must cover the canonical dataset exactly")
        for rid, reference in reference_units.items():
            validate_identity(rid, reference)
            if (reference.get("kind") not in {"scalar", "categorical", "qualitative", "mixed_target"}
                    or (reference["kind"] == "scalar" and not reference.get("unit"))):
                raise ValueError(f"Invalid FinanceBench reference type: {rid}")
        instance = cls()
        instance.root, instance.dataset, instance.records, instance.review = root, dataset, records, review
        instance.dataset_path = dataset_path
        instance.provenance = {"scorer": "verifiqa.eval.financebench_scoring",
            "review_path": str(path.relative_to(root)), "review_sha256": _sha(raw),
            "dataset_path": str(dataset_path.relative_to(root)), "dataset_sha256": _sha(dataset_bytes),
            "reviewed_references": len(records), "dataset_questions": len(dataset),
            "source_reviewed_is_post_hoc": True, "candidate_values_and_units_preserved": True}
        return instance

    def matches_dataset(self, path):
        path = Path(path).resolve()
        return path.is_relative_to(self.root) and path.is_file() and _sha(path.read_bytes()) == self.provenance["dataset_sha256"]

    def authority_audit(self, result_path):
        """Report run-specific witness defects separately from answer grading."""
        unavailable = {"status": "not_audited", "matching_snapshot": False,
                       "known_grounding_defects": None, "findings": [],
                       "display_note": "Grounding defects have not been audited for this run."}
        audit = self.review.get("authority_audit")
        if not audit:
            return {**unavailable, "reason": "authority_review_unavailable"}
        try:
            result_path = _repo_path(self.root, result_path)
            result_sha = _sha(result_path.read_bytes())
            if result_sha != audit["results_sha256"]:
                return {**unavailable, "reason": "result_hash_mismatch", "results_sha256": result_sha}
            audit_path = _repo_path(self.root, audit["path"])
            audit_bytes = audit_path.read_bytes()
            if _sha(audit_bytes) != audit["sha256"]:
                return {**unavailable, "reason": "authority_review_hash_mismatch"}
            findings = []
            for rid, record in self.records.items():
                for finding in record.get("authority_findings", []):
                    if (finding["results_sha256"] != result_sha
                            or finding["audit_sha256"] != audit["sha256"]
                            or _sha(_repo_path(self.root, finding["instance_path"]).read_bytes())
                                != finding["instance_sha256"]):
                        return {**unavailable, "reason": "authority_evidence_hash_mismatch"}
                    findings.append({"id": rid, **deepcopy(finding)})
            summary = json.loads(audit_bytes)["review_summary"]
            if (len(findings) != summary["confirmed_wrong_subset_authority_failures"]
                    or {finding["id"] for finding in findings}
                        != set(summary["wrong_subset_authority_failure_ids"])):
                return {**unavailable, "reason": "incomplete_authority_review_findings"}
            return {"status": "audited", "matching_snapshot": True,
                    "known_grounding_defects": len(findings), "findings": findings,
                    "display_note": f"{len(findings)} confirmed grounding defect(s) in this exact audited run; answer grading is separate.",
                    "results_sha256": result_sha, "audit_sha256": audit["sha256"],
                    "audit_path": audit["path"], "reason": "exact_run_and_evidence_hashes_match"}
        except (OSError, KeyError, ValueError) as exc:
            return {**unavailable, "reason": "authority_evidence_unavailable", "detail": str(exc)}

    def _canonical(self, row):
        rid = row.get("id") or row.get("financebench_id")
        canonical = self.dataset.get(rid)
        if canonical is None:
            raise ValueError(f"Unknown FinanceBench ID: {rid}")
        if row.get("question") != canonical["question"]:
            raise ValueError(f"FinanceBench question drift: {rid}")
        for key in ("gold_answer", "raw_answer"):
            if key in row and str(row[key]) != str(canonical["answer"]):
                raise ValueError(f"FinanceBench gold drift: {rid}")
        return rid, canonical

    def score_row(self, row, view=DEFAULT_SCORING_VIEW):
        if view not in SCORING_VIEWS:
            raise ValueError(f"Unknown FinanceBench scoring view: {view}")
        rid, canonical = self._canonical(row)
        reference = self.records.get(rid, {})
        reference_type = self.review["reference_units"][rid]
        comparison_row, unit_interpretation = _comparison_row(row, reference_type)
        record = reference if view == "source_reviewed" else {}
        ambiguous = record.get("kind") == "ambiguous_reference_scope"
        targets = record.get("plausible_targets" if ambiguous else "targets")
        if targets is None:
            targets = ([{"value": canonical["answer"],
                         "unit": reference.get("benchmark_unit", reference_type.get("unit", ""))}]
                       if reference_type["kind"] == "scalar" and not reference_type.get(
                           "reference_representation", "").startswith("qualitative_") else [])
        policy = {**self.review.get("scoring_policy", {}), **reference.get("scoring_policy", {})}
        absolute = policy.get("absolute_floor", 1.0)
        relative = policy.get("relative_tolerance", 0.01)
        comparisons, correctness = [], []
        for target in targets:
            unit = target.get("unit", "")
            conversion = None
            if unit:
                conversion = {"factor": 1.0, "comparison_unit": unit,
                    "absolute_tolerance": target.get("absolute_tolerance", absolute),
                    "tolerance_unit": target.get("tolerance_unit", policy.get("comparison_unit",
                                                       reference.get("comparison_unit", unit)))}
            comparisons.append(numeric_claim_comparison(comparison_row, str(target["value"]), gold_scale=unit,
                                                        reviewed_conversion=conversion))
            correctness.append(candidate_matches_gold(comparison_row, str(target["value"]), gold_scale=unit,
                reviewed_conversion=conversion, absolute_floor=absolute,
                relative_tolerance=relative, compare_magnitudes=False))
        correct = True if any(value is True for value in correctness) else (
            None if not correctness or all(value is None for value in correctness) else False)
        if ambiguous and correct is True:
            correct = None
        status = row.get("status")
        if status == "VERIFIED":
            outcome = "UNSCORED_VERIFIED" if correct is None else "TA" if correct else "FA"
        elif status == "VIOLATED":
            outcome = "UNSCORED_REJECTED" if correct is None else "FR" if correct else "TR"
        else:
            outcome = "AB"
        comparison = comparisons[0] if comparisons else {}
        stored_claim = claim_quantity(row)
        reason = ("ambiguous_reference_scope" if ambiguous and correct is None
                  else "outside_plausible_targets" if ambiguous else
                  "candidate_ungradeable" if correct is None else record.get("reason", "original_reference"))
        return {"id": rid, "view": view, "candidate_correct": correct, "correctness": correct,
            "outcome": outcome, "reason": reason, "reference_kind": record.get("kind", "benchmark"),
            f"{view}_candidate_correct": correct, f"{view}_outcome": outcome,
            "reference_targets": deepcopy(targets), "raw_answer": canonical["answer"],
            "comparison_unit": record.get("comparison_unit", comparison.get("gold_unit", "")),
            "candidate_value": comparison.get("candidate_value"),
            "claim_unit": stored_claim["unit"], "unit_interpretation": unit_interpretation,
            "comparison_claim_unit": comparison.get("claim_unit", ""),
            "claim_unit_source": comparison.get("claim_unit_source", ""),
            "comparison": comparison, "comparisons": comparisons}

    def annotate_row(self, row, view=DEFAULT_SCORING_VIEW):
        result = deepcopy(row)
        score = self.score_row(row, view)
        correct, status = score["candidate_correct"], row.get("status")
        decision_correct = ((correct if status == "VERIFIED" else not correct)
                            if correct is not None and status in {"VERIFIED", "VIOLATED"} else None)
        reason = (score["reason"] if correct is None else "no_solver_verdict"
                  if status not in {"VERIFIED", "VIOLATED"} else "solver_status_matches_reference"
                  if decision_correct else "solver_status_differs_from_reference")
        result.update({"answer_matches_gold": correct, "financebench_scoring": score,
            f"{view}_candidate_correct": correct,
            f"{view}_outcome": score["outcome"],
            "verified_correct": correct if status == "VERIFIED" else None,
            "solver_correct": decision_correct, "decision_correct": decision_correct,
            "solver_correct_reason": reason, "decision_correct_reason": reason,
            "actual_solver_status": {"VERIFIED": "SAT", "VIOLATED": "UNSAT"}.get(status, "ABSTAIN"),
            "expected_solver_status": "SAT" if correct is True else "UNSAT" if correct is False else None,
            "verifier_decision": {"VERIFIED": "accept", "VIOLATED": "reject"}.get(status, "abstain")})
        return result

    def score_rows(self, rows, view=DEFAULT_SCORING_VIEW):
        rows = list(rows)
        _index(rows, "id", "result")
        return [self.score_row(row, view) for row in rows]

    def summarize_rows(self, rows, elapsed_s=0, view=DEFAULT_SCORING_VIEW):
        rows = list(rows)
        _index(rows, "id", "result")
        scores = [self.score_row(row, view) for row in rows]
        counts = Counter(score["outcome"] for score in scores)
        ta, fa, tr, fr, ab, uv, ur = [counts[key] for key in
            ("TA", "FA", "TR", "FR", "AB", "UNSCORED_VERIFIED", "UNSCORED_REJECTED")]
        verified, violated = ta + fa + uv, tr + fr + ur
        total, scored = len(rows), ta + fa + tr + fr
        precision = ta / (ta + fa) if ta + fa else None
        accuracy = (ta + tr) / scored if scored else None
        lower, upper = ((ta / verified, (ta + uv) / verified) if verified else (None, None))
        return {"total": total, "verified": verified, "violated": violated, "abstain": ab,
            "TA": ta, "FA": fa, "TR": tr, "FR": fr, "AB": ab,
            "UNSCORED_VERIFIED": uv, "UNSCORED_REJECTED": ur,
            "unscored_verified": uv, "unscored_rejected": ur, "unscored_decisions": uv + ur,
            "scored_decisions": scored, "raw_decisions": verified + violated,
            "all_verifications": verified + violated, "solver_verdicts": verified + violated,
            "correct_verifications": ta + tr, "correct_solver_verdicts": ta + tr,
            "incorrect_verifications": fa + fr, "incorrect_solver_verdicts": fa + fr,
            "verified_correct": ta, "verified_incorrect": fa,
            "violated_correct": tr, "violated_incorrect": fr,
            "precision": precision, "verification_precision": precision, "sat_precision": precision,
            "accuracy": accuracy, "decision_accuracy": accuracy, "solver_accuracy": accuracy,
            "coverage": (verified + violated) / total if total else None,
            "scored_coverage": scored / total if total else None,
            "precision_lower_bound": lower, "precision_upper_bound": upper,
            "precision_bounds": [lower, upper], "elapsed_s": elapsed_s,
            "abstain_reasons": dict(Counter(row.get("failure_reason", "") for row in rows
                                             if row.get("status") not in {"VERIFIED", "VIOLATED"})),
            "scoring": {**self.provenance, "primary_view": view,
                        "graded_precision_denominator": ta + fa,
                        "graded_accuracy_denominator": scored}}
