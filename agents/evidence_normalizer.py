# agents/evidence_normalizer.py
from __future__ import annotations

import re
import math
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .types import Passage, Citation

# Optional settings import with safe fallbacks
try:
    from config.settings import settings
    _RECENCY_YEARS = getattr(settings, "EVIDENCE_RECENCY_YEARS", 10)
    _ALLOWLIST = set(getattr(settings, "ALLOWLIST_DOMAINS", []))
except Exception:
    _RECENCY_YEARS = 10
    _ALLOWLIST = set([
        "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "clinicaltrials.gov",
        "dailymed.nlm.nih.gov", "cdc.gov", "who.int", "cochranelibrary.com", "fda.gov"
    ])

PMID_RE = re.compile(r"\bPMID[:\s]*([0-9]{5,9})\b", re.I)
DOI_RE  = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Expanded signals for study type detection
_STUDY_SIGNALS = [
    ("systematic review", "MetaAnalysis", 0.95),
    ("meta-analysis", "MetaAnalysis", 1.00),
    ("practice guideline", "Guideline", 0.95),
    ("guideline", "Guideline", 0.92),
    ("randomized", "RCT", 0.90),
    ("double-blind", "RCT", 0.90),
    ("placebo-controlled", "RCT", 0.88),
    ("clinical trial", "Trial", 0.80),
    ("cohort", "Cohort", 0.75),
    ("case-control", "Cohort", 0.70),
    ("observational", "Cohort", 0.68),
    ("case series", "CaseReport", 0.45),
    ("case report", "CaseReport", 0.40),
    ("label", "Label", 0.90),  # FDA/label info (DailyMed)
    ("dailymed", "Label", 0.90),
    ("preprint", "Preprint", 0.50),
]

def _classify_study_type(text: str, meta: dict) -> (Optional[str], float):
    # Prefer explicit metadata if present
    mtype = (meta.get("study_type") or "").strip()
    if mtype:
        lowered = mtype.lower()
        for kw, label, score in _STUDY_SIGNALS:
            if kw in lowered:
                return label, score
        # fallback if metadata has something but didn't match our list
        return mtype, 0.7

    # Otherwise infer from text
    t = (text or "").lower()
    for kw, label, score in _STUDY_SIGNALS:
        if kw in t:
            return label, score
    return None, 0.6  # default mid confidence

def _coalesce_pmid(meta: dict, txt: str) -> Optional[str]:
    pmid = meta.get("pmid")
    if pmid:
        return str(pmid)
    m = PMID_RE.search(txt or "")
    return m.group(1) if m else None

def _coalesce_doi(meta: dict, txt: str) -> Optional[str]:
    doi = meta.get("doi")
    if doi:
        return doi
    m = DOI_RE.search(txt or "")
    return m.group(0) if m else None

def _coalesce_year(meta: dict, txt: str) -> Optional[int]:
    for k in ("year", "publication_year", "pub_year"):
        y = meta.get(k)
        if y:
            try:
                return int(str(y)[:4])
            except Exception:
                pass
    m = YEAR_RE.search(txt or "")
    return int(m.group(0)) if m else None

def _domain_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return None

def _pubmed_url_from_pmid(pmid: Optional[str]) -> Optional[str]:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

class EvidenceNormalizer:
    def normalize(self, claim_passages: Dict[str, List[Passage]]) -> List[Citation]:
        citations: List[Citation] = []
        # We cannot know "current year" reliably without system time; rely on metadata or recency window only.
        for cid, passages in claim_passages.items():
            for p in passages:
                txt = p["text"]
                meta = p.get("metadata", {}) or {}

                pmid = _coalesce_pmid(meta, txt)
                doi  = _coalesce_doi(meta, txt)
                year = _coalesce_year(meta, txt)
                study_type, base = _classify_study_type(txt, meta)

                # Choose/bake URL
                url = p.get("source") or meta.get("url") or _pubmed_url_from_pmid(pmid)

                # Venue/Journal if available
                venue = meta.get("venue") or meta.get("journal")

                # Start with base score from study type
                strength = base

                # Domain trust boost
                dom = _domain_from_url(url)
                if dom and dom in _ALLOWLIST:
                    strength += 0.05

                # Recency adjustment (soft): if year is within recency window, small boost;
                # if clearly older than window, small penalty.
                if year:
                    # assume current-ish year ~ now; we just scale to a rough window
                    # (no dependency on system date for deterministic tests)
                    # heuristic: >2000 baseline; bump if >= (2000 + (25 - recency years)) roughly
                    if year >= 2018:
                        strength += 0.05
                    if year >= 2022:
                        strength += 0.05
                    if year <= 2000:
                        strength -= 0.05
                    # If recency window is tight and year is way older, penalize mildly
                    if _RECENCY_YEARS <= 7 and year < 2010:
                        strength -= 0.05

                strength = _clamp01(strength)

                citations.append(Citation(
                    claim_id=cid,
                    passage_text=(txt or "")[:1000],
                    pmid=pmid,
                    doi=doi,
                    year=year,
                    study_type=study_type,
                    venue=venue,
                    url=url,
                    strength=strength,
                ))

        # Dedupe: prefer keys in this order pmid > doi > url > (doc_id,chunk_id)
        seen = set()
        unique: List[Citation] = []
        for c in citations:
            key = (c.get("claim_id"),
                   c.get("pmid") or "",
                   c.get("doi") or "",
                   c.get("url") or "",
                   (p.get("metadata", {}).get("doc_id") if isinstance(p, dict) else ""),
                   (p.get("metadata", {}).get("chunk_id") if isinstance(p, dict) else ""))
            # Since we don't have the original passage here, we stick with (claim_id, pmid, doi, url)
            key = (c.get("claim_id"), c.get("pmid") or "", c.get("doi") or "", c.get("url") or "")
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)

        # Sort citations by descending strength (useful downstream)
        unique.sort(key=lambda x: (x.get("strength") or 0.0), reverse=True)
        return unique
