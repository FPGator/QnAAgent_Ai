# agents/types.py
from typing import TypedDict, Literal, List, Optional, Dict, Any

class PICO(TypedDict, total=False):
    P: str
    I: str
    C: str
    O: str

ClaimType = Literal["efficacy", "safety", "dose", "diagnostic", "contraindication", "marketing"]
Certainty = Literal["may", "suggests", "is proven", "strongly", "insufficient"]

class Claim(TypedDict):
    id: str
    text: str
    type: ClaimType
    pico: PICO
    certainty: Certainty

class Passage(TypedDict, total=False):
    claim_id: str
    text: str
    source: Optional[str]
    metadata: Dict[str, Any]  # carries doc_id, chunk_id, venue, url, score, etc.

class Citation(TypedDict, total=False):
    claim_id: str
    passage_text: str
    pmid: Optional[str]
    doi: Optional[str]
    year: Optional[int]
    study_type: Optional[str]
    venue: Optional[str]
    url: Optional[str]
    strength: float  # 0.0–1.0 heuristic

VerdictStatus = Literal["Supported", "Partial", "Contradicted", "No Evidence"]

class Verdict(TypedDict, total=False):
    claim_id: str
    verdict: VerdictStatus
    citations: List[str]           # list of pmid/doi/url strings
    supporting_quotes: List[str]   # short verbatim snippets
    unsupported_bits: List[str]    # parts of the claim lacking support
    contradictions: List[str]      # contradictory findings
    rationale: str                 # brief explanation

class RiskFlag(TypedDict, total=False):
    code: Literal[
        "ABSOLUTE_CLAIM",
        "SUPPLEMENT_DISEASE_CLAIM",
        "DOSE_WITHOUT_SOURCE",
        "POPULATION_OVERGENERALIZATION",
        "OFF_LABEL",
        "MISSING_RISK_CONTEXT",
        "PHI_PII",
    ]
    where: str   # e.g., "claim_id=c_123" or "note"
    why: str     # short reason
    remedy: str  # concise fix instruction

__all__ = [
    "PICO",
    "ClaimType",
    "Certainty",
    "Claim",
    "Passage",
    "Citation",
    "VerdictStatus",
    "Verdict",
    "RiskFlag",
]
