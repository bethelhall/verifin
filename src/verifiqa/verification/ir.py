from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

from verifiqa.types import (
    CertificateFact,
    Claim,
    VerificationCertificate,
    VerificationFact,
    VerificationIR,
    VerificationSchema,
)


def ir_from_certificate(
    certificate: VerificationCertificate,
    claim: Claim,
    schema: VerificationSchema,
) -> VerificationIR:
    """Build the typed verifier IR used for SMT generation and validation."""

    facts_by_name: Dict[str, CertificateFact] = {fact.name: fact for fact in certificate.facts}
    facts = {}
    for name in schema.required_evidence:
        cert_fact = facts_by_name.get(name)
        value = float(claim.variables[name])
        if cert_fact is None:
            facts[name] = VerificationFact(name=name, value=value, unit=schema.fact_units.get(name, ""))
            continue
        facts[name] = VerificationFact(
            name=name,
            value=value,
            unit=cert_fact.unit,
            fact_type=cert_fact.fact_type,
            raw_value=cert_fact.raw_value,
            raw_unit=cert_fact.raw_unit,
            source_scale=cert_fact.source_scale,
            source_scale_quote=cert_fact.source_scale_quote,
            source_quote=cert_fact.source_quote,
            chunk_id=cert_fact.chunk_id,
            period=cert_fact.period,
            row_label=cert_fact.row_label,
            column=cert_fact.column,
            absence_scope=dict(cert_fact.absence_scope),
        )

    return VerificationIR(
        metric=schema.metric,
        formula=schema.formula,
        facts=facts,
        claimed_value=float(claim.claimed_value),
        claim_unit=schema.claim_unit,
        tolerance=float(schema.tolerance),
        computed_unit=schema.computed_unit,
        precision_digits=schema.precision_digits,
        tolerance_source=schema.tolerance_source,
        period=claim.period,
        query_type=schema.query_type,
    )


def ir_from_claim_schema(claim: Claim, schema: VerificationSchema) -> VerificationIR:
    facts = {
        name: VerificationFact(
            name=name,
            value=float(claim.variables[name]),
            unit=schema.fact_units.get(name, ""),
        )
        for name in schema.required_evidence
        if name in claim.variables
    }
    return VerificationIR(
        metric=schema.metric,
        formula=schema.formula,
        facts=facts,
        claimed_value=float(claim.claimed_value),
        claim_unit=schema.claim_unit,
        tolerance=float(schema.tolerance),
        computed_unit=schema.computed_unit,
        precision_digits=schema.precision_digits,
        tolerance_source=schema.tolerance_source,
        period=claim.period,
        query_type=schema.query_type,
    )


def claim_schema_from_ir(ir: VerificationIR) -> Tuple[Claim, VerificationSchema]:
    variables = sorted(ir.facts)
    computed = f"computed_{ir.metric}"
    claim = Claim(
        metric=ir.metric,
        claimed_value=float(ir.claimed_value),
        period=ir.period,
        variables={name: float(ir.facts[name].value) for name in variables},
    )
    schema = VerificationSchema(
        metric=ir.metric,
        formula=ir.formula,
        allowed_variables=variables + [computed],
        required_evidence=variables,
        allowed_operators=["+", "-", "*", "/", "=", "<=", ">=", "<", ">", "or", "and"],
        tolerance=float(ir.tolerance),
        unit_policy="ir_supplied",
        period_policy="ir_supplied",
        query_type=ir.query_type,
        fact_units={name: ir.facts[name].unit for name in variables},
        claim_unit=ir.claim_unit,
        computed_unit=ir.computed_unit or ir.claim_unit,
        precision_digits=ir.precision_digits,
        tolerance_source=ir.tolerance_source,
    )
    return claim, schema


def replace_claimed_value(ir: VerificationIR, claimed_value: float) -> VerificationIR:
    return replace(ir, claimed_value=float(claimed_value))
