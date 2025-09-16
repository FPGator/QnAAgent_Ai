# agents/verification_agent_openai.py
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple, Set

from openai import OpenAI
from .types import Verdict, VerdictStatus, Claim, Passage, Citation

# ----- settings with safe fallbacks -----
try:
    from config.settings import settings
    _MODEL = getattr(settings, "OPENAI_VERIFY_MODEL", "gpt-4o-mini")
    _TEMP = getattr(settings, "TEMPERATURE_VERIFY", 0.0)
    _MAXTOK = getattr(settings, "VERIFY_MAX_TOKENS", 400)
    _RETRY = max(0, getattr(settings, "LLM_RETRY_COUNT", 1))
    _KPASS = max(1, getattr(settings, "VERIFY_PASSAGES_PER_CLAIM", 3))
    _PASSAGE_CLIP = 900       # chars per passage in prompt
    _TOTAL_PASSAGES_CLIP = 3200  # total chars for all passages in prompt
except Exception:
    _MODEL = "gpt-4o-mini"
    _TEMP = 0.0
    _MAXTOK = 400
    _RETRY = 1
    _KPASS = 3
    _PASSAGE_CLIP = 900
    _TOTAL_PASSAGES_CLIP = 3200

logger = logging.getLogger(__name__)
_client = OpenAI()

_SYSTEM = (
    "You are a strict medical content verifier. "
    "For each claim, you will assess the provided passages and return ONLY valid JSON "
    "that matches the exact schema. Do not include extra prose."
)

# Single-claim JSON schema (string to embed in the prompt)
_JSON_SCHEMA = r'''
{
  "claim_id": "<string>",
  "verdict": "Supported|Partial|Contradicted|No Evidence",
  "citations": ["<pmid|doi|url>", "..."],
  "supporting_quotes": ["<verbatim snippet>", "..."],
  "unsupported_bits": ["<which part of claim lacked support>", "..."],
  "contradictions": ["<brief contradictory finding>", "..."],
  "rationale": "<short explanation>"
}
'''

_USER_TMPL = """
You will verify ONE medical claim using the passages and citation hints below.

Rules:
- Judge ONLY from the given material. If evidence is weak or mixed, use "Partial".
- If the best evidence contradicts the claim, use "Contradicted".
- If there is no relevant support in the passages, use "No Evidence".
- Prefer higher quality evidence (meta-analysis/guideline > RCT > cohort > case report).
- Use short verbatim quotes for support (<= 25 words each; max 3 quotes).
- In "citations", choose ONLY from the allowed set provided; list up to 5 identifiers (pmid/doi/url).
- Output STRICT JSON only (no markdown, no commentary) matching this schema:
{json_schema}

Claim:
{claim_text}

Passages (top-{k}, clipped):
{passages_txt}

Allowed citations (choose only from these; if none apply, return []):
{allowed_citation_list}

Citation hints (metadata for context; may overlap with passages):
{citation_hints}
""".strip()


def _extract_json(payload: str) -> str:
    """Robustly extract the first top-level JSON object from a string."""
    m = re.search(r"\{.*\}", payload, flags=re.S)
    return m.group(0) if m else payload.strip()


def _clip(s: str, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[:n]


def _build_allowed_citation_set(citations: List[Citation], claim_id: str) -> Set[str]:
    allowed: Set[str] = set()
    for c in citations:
        if c.get("claim_id") != claim_id:
            continue
        if c.get("pmid"):
            allowed.add(str(c["pmid"]))
        if c.get("doi"):
            allowed.add(str(c["doi"]))
        if c.get("url"):
            allowed.add(str(c["url"]))
    return allowed


class VerificationAgent:
    """
    Per-claim Verification Agent.

    Public API:
      - verify(claims, claim_passages, citations, k_passages=3) -> List[Verdict]
      - summarize(verdicts) -> str
    """

    def __init__(
        self,
        model: str = _MODEL,
        temperature: float = _TEMP,
        max_tokens: int = _MAXTOK,
    ):
        self.client = _client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_passages_block(self, passages: List[Passage], k: int) -> str:
        lines = []
        total = 0
        for i, p in enumerate(passages[:k], start=1):
            snippet = _clip(p.get("text", ""), _PASSAGE_CLIP)
            src = p.get("source") or (p.get("metadata") or {}).get("source") or ""
            block = f"[P{i}] {snippet}\n-- source: {src}".strip()
            # keep overall prompt tiny
            if total + len(block) > _TOTAL_PASSAGES_CLIP:
                break
            lines.append(block)
            total += len(block)
        return "\n\n".join(lines) if lines else "(no passages)"

    def _build_citation_hints(self, citations: List[Citation], claim_id: str, max_items: int = 6) -> str:
        rows = []
        n = 0
        for c in citations:
            if c.get("claim_id") != claim_id:
                continue
            tag = c.get("pmid") or c.get("doi") or c.get("url") or "unknown"
            yr = c.get("year")
            st = c.get("study_type") or "NA"
            rows.append(f"- {tag} (year={yr}, type={st}, strength={c.get('strength')})")
            n += 1
            if n >= max_items:
                break
        return "\n".join(rows) if rows else "(none)"

    def _verify_one(self, claim: Claim, passages: List[Passage], citations: List[Citation], k_passages: int) -> Verdict:
        allowed_set = _build_allowed_citation_set(citations, claim["id"])
        allowed_list_str = ", ".join(sorted(allowed_set)) if allowed_set else "(none)"

        user = _USER_TMPL.format(
            json_schema=_JSON_SCHEMA,
            claim_text=claim["text"],
            k=k_passages,
            passages_txt=self._build_passages_block(passages, k_passages),
            allowed_citation_list=allowed_list_str,
            citation_hints=self._build_citation_hints(citations, claim["id"]),
        )

        attempts = 1 + _RETRY
        raw = ""
        last_err = None
        for i in range(attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": user if i == 0 else user + "\n\nReturn ONLY the JSON object."},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw = (resp.choices[0].message.content or "").strip()
                break
            except Exception as e:
                last_err = e
                logger.warning(f"OpenAI verify attempt {i+1}/{attempts} failed: {e}")

        if not raw:
            logger.error(f"OpenAI verification failed: {last_err}")
            return Verdict(
                claim_id=claim["id"],
                verdict="No Evidence",
                citations=[],
                supporting_quotes=[],
                unsupported_bits=["Model error or timeout."],
                contradictions=[],
                rationale="Model error.",
            )

        try:
            payload = _extract_json(raw)
            data = json.loads(payload)
            v: Verdict = Verdict(
                claim_id=data.get("claim_id") or claim["id"],
                verdict=data.get("verdict") or "No Evidence",
                citations=list(data.get("citations") or []),
                supporting_quotes=list(data.get("supporting_quotes") or []),
                unsupported_bits=list(data.get("unsupported_bits") or []),
                contradictions=list(data.get("contradictions") or []),
                rationale=data.get("rationale") or "",
            )
        except Exception as e:
            logger.warning(f"JSON parse failed; marking 'No Evidence'. Raw head: {raw[:200]}")
            return Verdict(
                claim_id=claim["id"],
                verdict="No Evidence",
                citations=[],
                supporting_quotes=[],
                unsupported_bits=["Invalid JSON returned by model."],
                contradictions=[],
                rationale="Parser failed.",
            )

        # ---- normalize & validate verdict ----
        allowed: Tuple[VerdictStatus, ...] = ("Supported", "Partial", "Contradicted", "No Evidence")
        if v["verdict"] not in allowed:
            v["verdict"] = "No Evidence"

        # Filter citations to allowed set (prevents hallucinated identifiers)
        if allowed_set:
            v["citations"] = [c for c in (v.get("citations") or []) if c in allowed_set]
        else:
            v["citations"] = []

        # Clean up quotes: <= 25 words each, max 3 quotes
        cleaned_quotes: List[str] = []
        for q in v.get("supporting_quotes", [])[:3]:
            words = (q or "").split()
            if len(words) > 25:
                q = " ".join(words[:25])
            if q.strip():
                cleaned_quotes.append(q.strip())
        v["supporting_quotes"] = cleaned_quotes

        # Dedupe lists
        v["citations"] = list(dict.fromkeys(v.get("citations", [])))
        v["unsupported_bits"] = list(dict.fromkeys(v.get("unsupported_bits", [])))
        v["contradictions"] = list(dict.fromkeys(v.get("contradictions", [])))

        return v

    def verify(
        self,
        claims: List[Claim],
        claim_passages: Dict[str, List[Passage]],
        citations: List[Citation],
        k_passages: int = _KPASS,
    ) -> List[Verdict]:
        verdicts: List[Verdict] = []
        for c in claims:
            p = claim_passages.get(c["id"], []) or []
            v = self._verify_one(c, p, citations, k_passages=k_passages)
            verdicts.append(v)
        return verdicts

    @staticmethod
    def summarize(verdicts: List[Verdict]) -> str:
        counts = {"Supported": 0, "Partial": 0, "Contradicted": 0, "No Evidence": 0}
        for v in verdicts:
            label = v.get("verdict", "No Evidence")
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values()) or 0
        head = (
            f"Claims verified: {total}.  Supported: {counts['Supported']}, "
            f"Partial: {counts['Partial']}, Contradicted: {counts['Contradicted']}, "
            f"No Evidence: {counts['No Evidence']}."
        )
        highlights = []
        for v in verdicts[:5]:
            cid = v["claim_id"]
            ver = v["verdict"]
            cites = ", ".join(v.get("citations", [])[:2]) or "no citations"
            highlights.append(f"{cid}: {ver} ({cites})")
        tail = " | " + " ; ".join(highlights) if highlights else ""
        return head + tail
