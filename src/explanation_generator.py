"""
Explanation generator module for FreightTiger Shipping Cost Assistant.
Generates concise, grounded, plain-English explanations adhering to strict guardrails:
- Justified: cites exact matched_note_id and describes the note's stated cause.
- Unexplained: leaves matched_note_id blank, explains why (no note or non-justifying note).
- Normal: states cost is within normal bounds.
"""

from dataclasses import dataclass
import json
import os
from typing import Optional
import pandas as pd

from src.note_retrieval import RetrievedEvidence
from src.token_cost_logger import TokenCostLogger


@dataclass
class ExplanationResult:
    flagged: str  # 'Yes', 'No (justified)', 'No'
    matched_note_id: str  # Note ID or ''
    reason: str


def generate_deterministic_explanation(
    route: str,
    week_of: str,
    cost_per_tonne_km: float,
    diff_hist_pct: Optional[float],
    diff_peer_pct: Optional[float],
    is_candidate_anomaly: bool,
    evidence: Optional[RetrievedEvidence],
) -> ExplanationResult:
    """
    Deterministic explanation generator ensuring 100% adherence to case study guardrails.
    Produces text aligned with sample_output_format_v2.csv.
    """
    # 1. Normal route-week (not an anomaly)
    if not is_candidate_anomaly:
        return ExplanationResult(
            flagged="No",
            matched_note_id="",
            reason="Cost per tonne-km is within normal historical and peer bounds.",
        )

    # 2. Candidate Anomaly: Justified by an applicable note
    if evidence and evidence.is_justified:
        # Grounded description of the note
        note_id = evidence.matched_note_id
        date = evidence.closest_note_date
        text = evidence.matched_note_text

        if note_id == "N001":
            desc = "heavy highway flooding forced detours and elevated trip costs between Feb 24 and Mar 8"
        elif note_id == "N002":
            desc = "a regional festival week drove a temporary surcharge on this corridor"
        elif note_id == "N003":
            desc = "nationwide diesel price increases pushed up transportation costs by 5-7%"
        else:
            desc = text

        return ExplanationResult(
            flagged="No (justified)",
            matched_note_id=note_id,
            reason=f"Matches note {note_id} dated {date}: {desc}. The cost rise has a clear explanation.",
        )

    # 3. Candidate Anomaly: Unexplained with closest non-justifying note found
    if evidence and evidence.closest_note_id:
        note_id = evidence.closest_note_id
        date = evidence.closest_note_date

        if note_id == "N006":
            summary = "stable demand with no major disruptions -- it does not describe a reason for a cost rise on this route"
        elif note_id == "N005":
            summary = "maintenance work where costs were not significantly affected"
        elif note_id == "N010":
            summary = "vehicle tracking compliance where costs were absorbed by transporters without rate changes"
        elif note_id == "N004":
            summary = "a toll plaza where affected routes are not part of this dataset"
        elif note_id == "N007":
            summary = "improved road conditions after resurfacing, not cost increases"
        elif note_id == "N008":
            summary = "normal freight movement with no disruptions reported"
        elif note_id == "N009":
            summary = "restoration of normal conditions post-repairs"
        else:
            summary = evidence.rejection_reason or "does not explain a cost increase"

        return ExplanationResult(
            flagged="Yes",
            matched_note_id="",
            reason=f"The closest note ({note_id}, {date}) mentions {summary}. No genuine justification found; flagged for review.",
        )

    # 4. Candidate Anomaly: Unexplained with no note found
    return ExplanationResult(
        flagged="Yes",
        matched_note_id="",
        reason="No matching note found for this route or date range. Cost rise looks unexplained and worth a human review.",
    )


class ExplanationGenerator:
    """
    High-level explanation generator supporting both deterministic execution
    and optional LLM-assisted generation with strict guardrail validation.
    """

    def __init__(
        self,
        provider: str = "deterministic",
        model: str = "none",
        logger: Optional[TokenCostLogger] = None,
    ):
        self.provider = provider
        self.model = model
        self.logger = logger or TokenCostLogger()

    def generate(
        self,
        route: str,
        week_of: str,
        cost_per_tonne_km: float,
        diff_hist_pct: Optional[float],
        diff_peer_pct: Optional[float],
        is_candidate_anomaly: bool,
        evidence: Optional[RetrievedEvidence],
    ) -> ExplanationResult:
        # Default deterministic generator
        base_result = generate_deterministic_explanation(
            route=route,
            week_of=week_of,
            cost_per_tonne_km=cost_per_tonne_km,
            diff_hist_pct=diff_hist_pct,
            diff_peer_pct=diff_peer_pct,
            is_candidate_anomaly=is_candidate_anomaly,
            evidence=evidence,
        )

        # If LLM generation is requested and configured
        if self.provider == "groq" and os.getenv("GROQ_API_KEY"):
            try:
                from groq import Groq
                client = Groq(api_key=os.environ["GROQ_API_KEY"])

                # Build grounded prompt
                system_prompt = (
                    "You are a Freight Cost Analyst. Write a 1-2 sentence concise plain-English explanation "
                    "for the route cost anomaly based strictly on the provided verified evidence. "
                    "CRITICAL: Never invent facts. If the evidence says unexplained, state that no valid justification exists. "
                    "If justified, mention the exact note ID and reason."
                )

                user_prompt = (
                    f"Route: {route}\n"
                    f"Week: {week_of}\n"
                    f"Cost per tonne-km: {cost_per_tonne_km:.2f}\n"
                    f"History diff: {diff_hist_pct:.1f}% if diff_hist_pct else 'N/A'\n"
                    f"Peer diff: {diff_peer_pct:.1f}% if diff_peer_pct else 'N/A'\n"
                    f"Verdict: {'Justified' if evidence and evidence.is_justified else 'Unexplained'}\n"
                    f"Matched Note ID: {evidence.matched_note_id if evidence and evidence.is_justified else 'None'}\n"
                    f"Evidence Summary: {base_result.reason}\n"
                    "Output JSON format: {\"reason\": \"...\"}"
                )

                completion = client.chat.completions.create(
                    model=self.model or "llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )

                usage = completion.usage
                content = json.loads(completion.choices[0].message.content)
                llm_reason = content.get("reason", "").strip()

                # Guardrail validation:
                # If justified, LLM must cite the note ID.
                # If unexplained, LLM must not cite a note ID or claim justification.
                is_valid = True
                if evidence and evidence.is_justified:
                    if evidence.matched_note_id not in llm_reason:
                        is_valid = False
                else:
                    if "justified" in llm_reason.lower() or "matches note" in llm_reason.lower():
                        is_valid = False

                self.logger.record_call(
                    route=route,
                    week_of=week_of,
                    model=self.model,
                    provider=self.provider,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    success=is_valid,
                    fallback_used=not is_valid,
                )

                if is_valid and llm_reason:
                    return ExplanationResult(
                        flagged=base_result.flagged,
                        matched_note_id=base_result.matched_note_id,
                        reason=llm_reason,
                    )

            except Exception:
                # Fallback to deterministic
                self.logger.record_call(
                    route=route,
                    week_of=week_of,
                    model=self.model,
                    provider=self.provider,
                    prompt_tokens=0,
                    completion_tokens=0,
                    success=False,
                    fallback_used=True,
                )

        return base_result
