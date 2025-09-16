# agents/safety_rewriter_openai.py
from __future__ import annotations

import logging
import re
from typing import List, Set

from openai import OpenAI
from .types import Claim, Verdict, Citation, RiskFlag

# Optional settings with safe fallbacks
try:
    from config.settings import settings
    _MODEL = getattr(settings, "OPENAI_REWRITER_MODEL", "gpt-4o-mini")
    _TEMP = getattr(settings, "TEMPERATURE_REWRITE", 0.2)
    _MAXTOK = getattr(settings, "REWRITE_MAX_TOKENS", 900)
    _RETRY = getattr(settings, "LLM_RETRY_COUNT", 1)
    _DISCLAIMER = getattr(settings, "DISCLAIMER_TEXT",
                          "This note is informational and not a substitute for clinical judgment or patient-specific guidance.")
    _NOTE_MAX_CHARS = 8000
except Exception:
    _MODEL = "gpt-4o-mini"
    _TEMP = 0.2
    _MAXTOK = 900
    _RETRY = 1
    _DISCLAIMER = "This note is informational and not a substitute for clinical judgment or patient-specific guidance."
    _NOTE_MAX_CHARS = 8000

logger = logging.getLogger(__name__)
_client = OpenAI()

SYSTEM = (
    "You are a medical content editor. Rewrite clinician notes for safety, clarity, and compliance. "
    "Be conservative and evidence-anchored. Keep it concise and clinically useful."
)

USER_TMPL = """
Rewrite the note following these rules:

1) Respect evidence:
   - For Supported claims: keep, but prefer qualified language (e.g., "may improve").
   - For Partial: keep with stronger qualifiers and scope (population), and mention uncertainty.
   - For No Evidence/Contradicted: remove or present as unsubstantiated; avoid recommending actions.

2) Scope:
   - Add/retain population (P) if missing (e.g., "in adults with X").
   - Avoid overgeneralizations ("for everyone", "always").

3) Risk & compliance:
   - Address risk flags below; fix language accordingly.
   - For dosing, include only if the claim's verdict is Supported; otherwise qualify or omit.
   - Avoid supplement disease-treatment claims; use structure/function language if needed.
   - Do not add new clinical claims beyond those in the input.

4) Citations:
   - Insert inline markers ONLY from the allowed list below, like [PMID:12345678] or [DOI:10.xxxx/yyy] after supported statements (max 1–2 per claim).
   - Do NOT invent new citations or formats.

5) Tone & readability:
   - Plain-language where possible (8th–10th grade).
   - Keep the original intent but reduce liability.

6) End with this one-line disclaimer exactly:
   - "{disclaimer}"

Original Note:
{note}

Claims (with verdicts):
{claims_block}

Supported claim IDs:
{supported_ids}

Allowed citation tags (use only these; at most 1–2 per claim):
{allowed_tags}

Top Citations (reference for context; do not copy new tags from here):
{citations_block}

Risk Flags:
{risk_block}

Rewrite the full note. Output ONLY the rewritten note text.
""".strip()


def _claims_block(claims: List[Claim], verdicts: List[Verdict]) -> str:
    vmap = {v["claim_id"]: v for v in verdicts}
    lines = []
    for c in claims:
        v = vmap.get(c["id"], {"verdict": "No Evidence"})
        P = (c.get("pico") or {}).get("P", "")
        lines.append(f"- [{c['id']}] {c['text']}  | verdict={v.get('verdict')}  | P='{P}'")
    return "\n".join(lines) if lines else "(none)"


def _citations_block(citations: List[Citation]) -> str:
    seen = set()
    rows = []
    for c in citations:
        tag = c.get("pmid") or c.get("doi") or c.get("url")
        if not tag or tag in seen:
            continue
        seen.add(tag)
        yr = c.get("year")
        st = c.get("study_type") or "NA"
        rows.append(f"- {tag} (year={yr}, type={st}, strength={c.get('strength')})")
        if len(rows) >= 16:
            break
    return "\n".join(rows) or "(none)"


def _risk_block(risks: List[RiskFlag]) -> str:
    return "\n".join(
        f"- {r['code']} @ {r.get('where')}: {r.get('why')}. Remedy: {r.get('remedy')}"
        for r in risks
    ) or "(none)"


def _supported_ids(verdicts: List[Verdict]) -> str:
    ids = [v["claim_id"] for v in verdicts if v.get("verdict") == "Supported"]
    return ", ".join(ids) if ids else "(none)"


def _allowed_tags(citations: List[Citation]) -> Set[str]:
    allowed: Set[str] = set()
    for c in citations:
        if c.get("pmid"):
            allowed.add(f"PMID:{c['pmid']}")
        if c.get("doi"):
            allowed.add(f"DOI:{c['doi']}")
        if c.get("url"):
            allowed.add(f"URL:{c['url']}")
    return allowed


# strip any [PMID:/DOI:/URL:] markers that aren't in allowed set
_TAG_RE = re.compile(r"\[([A-Z]{3,4}):([^\]\n]+)\]")

def _filter_unapproved_markers(text: str, allowed: Set[str]) -> str:
    def repl(m: re.Match) -> str:
        tag = f"{m.group(1)}:{m.group(2)}"
        return f"[{tag}]" if tag in allowed else ""  # drop unapproved markers
    return _TAG_RE.sub(repl, text)


def _truncate(s: str, max_chars: int) -> str:
    return s if not s or len(s) <= max_chars else s[:max_chars]


class SafetyRewriter:
    def __init__(self, model: str = _MODEL, temperature: float = _TEMP, max_tokens: int = _MAXTOK):
        self.client = _client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call_llm(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,
        )
        return (resp.choices[0].message.content or "").strip()

    def rewrite(
        self,
        note: str,
        claims: List[Claim],
        verdicts: List[Verdict],
        citations: List[Citation],
        risks: List[RiskFlag],
    ) -> str:
        safe_note = _truncate(note or "", _NOTE_MAX_CHARS)
        allowed = _allowed_tags(citations)

        prompt = USER_TMPL.format(
            disclaimer=_DISCLAIMER,
            note=safe_note,
            claims_block=_claims_block(claims, verdicts),
            supported_ids=_supported_ids(verdicts),
            allowed_tags=", ".join(sorted(allowed)) if allowed else "(none)",
            citations_block=_citations_block(citations),
            risk_block=_risk_block(risks),
        )

        # try once (+ optional retry)
        attempts = 1 + max(0, _RETRY)
        out = ""
        last_err = None
        for i in range(attempts):
            try:
                out = self._call_llm(prompt if i == 0 else prompt + "\n\nReturn ONLY the rewritten note text.")
                break
            except Exception as e:
                last_err = e
                logger.warning(f"SafetyRewriter attempt {i+1}/{attempts} failed: {e}")

        if not out:
            logger.error(f"SafetyRewriter error: {last_err}")
            # fail-safe: return original note with disclaimer
            base = (safe_note or "").strip()
            return (base + ("\n\n" if base else "") + f"Disclaimer: {_DISCLAIMER}").strip()

        # Post-filter: drop any unapproved [PMID:/DOI:/URL:] markers
        out = _filter_unapproved_markers(out, allowed)

        # Ensure disclaimer present exactly once at the end
        if _DISCLAIMER not in out:
            if not out.endswith("\n"):
                out += "\n"
            out += _DISCLAIMER

        return out.strip()
