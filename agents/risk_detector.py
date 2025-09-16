# agents/risk_detector.py
import re
from typing import List
from .types import Claim, Verdict, RiskFlag

# Optional settings (safe fallbacks)
try:
    from config.settings import settings
    _ENABLE_PHI = getattr(settings, "ENABLE_PHI_CHECK", True)
except Exception:
    _ENABLE_PHI = True

ABSOLUTES = re.compile(r"\b(cure[sd]?|guarantee[sd]?|always|never|proven|works for everyone)\b", re.I)
DOSE_RE = re.compile(r"\b(\d+(\.\d+)?)\s*(mg|mcg|µg|ug|g|ml|mL|units|IU)\b", re.I)
OVERGEN = re.compile(r"\b(for everyone|all ages|anyone|everyone|works for all)\b", re.I)

PHI_PATTERNS = [
    re.compile(r"\bDOB[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I),
    re.compile(r"\bMRN[:\s]*[A-Z0-9\-]{6,}\b", re.I),
    re.compile(r"\b(?:\+?1[\s\-\.]?)?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b"),  # phone
    re.compile(r"\b\d{1,5}\s+[A-Za-z0-9.\- ]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr)\b", re.I),  # address-ish
]

SUPPLEMENT_WORDS = re.compile(
    r"\b(supplement|vitamin|herbal|herb|botanical|turmeric|curcumin|ashwagandha|berberine|omega[-\s]?3|fish oil|melatonin)\b",
    re.I,
)
DISEASE_WORDS = re.compile(
    r"\b(treat(?:s|ing|ment)?|cure(?:s|d|ing)?|prevent(?:s|ion|ing)?|mitigat(?:e|es|ing|ion)?)\b.*\b("
    r"cancer|diabetes|alzheimer(?:'s)?|covid(?:-19)?|sars[-\s]?cov[-\s]?2|hypertension|arthritis|depression|anxiety|asthma|migraine"
    r")\b",
    re.I,
)

class RiskyClaimDetector:
    """
    Deterministic rule-based checks that yield actionable RiskFlag items.
    You can extend this with a small LLM pass later if desired.
    """

    def detect(self, clinician_note: str, claims: List[Claim], verdicts: List[Verdict]) -> List[RiskFlag]:
        flags: List[RiskFlag] = []

        # --- PHI/PII in the note (configurable) ---
        if _ENABLE_PHI:
            for rx in PHI_PATTERNS:
                if rx.search(clinician_note or ""):
                    flags.append(RiskFlag(
                        code="PHI_PII",
                        where="note",
                        why="Note contains possible personally identifiable/health information.",
                        remedy="Remove identifiers (names, DOB, MRN, phone, addresses) or anonymize."
                    ))
                    break

        # Lookups
        verdict_by_claim = {v["claim_id"]: v for v in verdicts}
        has_safety_claim = any((c.get("type") == "safety") for c in claims)

        for c in claims:
            text = c.get("text", "") or ""
            cid = c["id"]
            v = verdict_by_claim.get(cid, {"verdict": "No Evidence"})
            verdict_label = v.get("verdict", "No Evidence")

            # ABSOLUTE_CLAIM: only flag if not fully Supported
            if ABSOLUTES.search(text) or c.get("certainty") in ("is proven", "strongly"):
                if verdict_label in ("Partial", "Contradicted", "No Evidence"):
                    flags.append(RiskFlag(
                        code="ABSOLUTE_CLAIM",
                        where=f"claim_id={cid}",
                        why=f"Definitive language but verdict={verdict_label}.",
                        remedy="Use qualified language (e.g., 'may help', 'associated with') and cite strongest evidence."
                    ))

            # SUPPLEMENT_DISEASE_CLAIM
            if SUPPLEMENT_WORDS.search(text) and DISEASE_WORDS.search(text):
                flags.append(RiskFlag(
                    code="SUPPLEMENT_DISEASE_CLAIM",
                    where=f"claim_id={cid}",
                    why="Supplement positioned to treat/cure/prevent a disease.",
                    remedy="Avoid disease-treatment claims for supplements; rephrase to structure/function or cite approved indications."
                ))

            # DOSE_WITHOUT_SOURCE
            if c.get("type") == "dose" or DOSE_RE.search(text):
                if verdict_label != "Supported":
                    flags.append(RiskFlag(
                        code="DOSE_WITHOUT_SOURCE",
                        where=f"claim_id={cid}",
                        why="Dosing information without strong supporting citation.",
                        remedy="Provide source (guideline/RCT/label) or remove/qualify dosing details."
                    ))

            # POPULATION_OVERGENERALIZATION (skip for pure 'safety' claims)
            P = (c.get("pico") or {}).get("P", "").strip()
            if c.get("type") in ("efficacy", "dose", "diagnostic", "marketing"):
                if OVERGEN.search(text) or not P:
                    flags.append(RiskFlag(
                        code="POPULATION_OVERGENERALIZATION",
                        where=f"claim_id={cid}",
                        why="Population not specified or overgeneralized.",
                        remedy="Name the intended population (e.g., 'adults with X'); limit scope to studied groups."
                    ))

            # OFF_LABEL (heuristic)
            if "off-label" in text.lower():
                flags.append(RiskFlag(
                    code="OFF_LABEL",
                    where=f"claim_id={cid}",
                    why="Off-label suggestion detected.",
                    remedy="State that use is off-label, disclose uncertainties, and cite authoritative guidance."
                ))

        # MISSING_RISK_CONTEXT (no safety statements at all)
        if not has_safety_claim:
            flags.append(RiskFlag(
                code="MISSING_RISK_CONTEXT",
                where="note",
                why="No safety/contraindication context found.",
                remedy="Add brief safety notes (common adverse effects, contraindications, monitoring) with citations."
            ))

        # De-duplicate and sort for stable output (code, then where)
        dedup = []
        seen = set()
        for f in flags:
            key = (f.get("code"), f.get("where"), f.get("why"))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(f)
        dedup.sort(key=lambda x: (x.get("code", ""), x.get("where", "")))
        return dedup
