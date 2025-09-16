# # agents/claim_extractor_openai.py
# import json
# import logging
# import re
# import uuid
# from typing import List

# from openai import OpenAI
# from .types import Claim

# # Optional settings import with safe fallbacks
# try:
#     from config.settings import settings
#     _MODEL = getattr(settings, "OPENAI_CLAIM_MODEL", "gpt-4o-mini")
#     _MAX_TOKENS = getattr(settings, "CLAIM_MAX_TOKENS", 600)
#     _TEMP = getattr(settings, "TEMPERATURE_CLAIM", 0.0)
#     _RETRY = getattr(settings, "LLM_RETRY_COUNT", 1)
#     _NOTE_MAX_CHARS = 8000  # soft cap; prompt is concise so this is safe
# except Exception:
#     _MODEL = "gpt-4o-mini"
#     _MAX_TOKENS = 600
#     _TEMP = 0.0
#     _RETRY = 1
#     _NOTE_MAX_CHARS = 8000

# logger = logging.getLogger(__name__)
# _client = OpenAI()

# SYSTEM = (
#     "You extract atomic medical claims from clinician notes. "
#     "Return STRICT JSON with key 'claims' as a list of claim objects. "
#     "Do NOT add prose."
# )

# USER_TMPL = """
# Extract atomic, auditable medical claims.

# Rules:
# - Split multi-part sentences into separate claims.
# - Preserve exact meaning; no new facts.
# - Fill PICO when possible; leave empty strings if unknown.
# - Classify type: efficacy, safety, dose, diagnostic, contraindication, marketing.
# - Certainty from author wording: may, suggests, is proven, strongly, insufficient.

# Return JSON ONLY:
# {"claims":[
#   {"id":"", "text":"", "type":"", "pico":{"P":"","I":"","C":"","O":""}, "certainty":""}
# ]}

# Note:
# {note}
# """.strip()


# def _first_json_object(text: str) -> str:
#     """Extract the first top-level JSON object from text (in case the model leaks text)."""
#     m = re.search(r"\{.*\}", text, flags=re.S)
#     return m.group(0) if m else text.strip()


# def _truncate(s: str, max_chars: int) -> str:
#     return s if len(s) <= max_chars else s[:max_chars]


# class ClaimExtractor:
#     def __init__(
#         self,
#         model: str = _MODEL,
#         max_tokens: int = _MAX_TOKENS,
#         temperature: float = _TEMP,
#     ):
#         self.client = _client
#         self.model = model
#         self.max_tokens = max_tokens
#         self.temperature = temperature

#     def _call_llm(self, note: str) -> str:
#         """Single LLM call; isolated for retry."""
#         messages = [
#             {"role": "system", "content": SYSTEM},
#             {"role": "user", "content": USER_TMPL.format(note=note)},
#         ]
#         resp = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=self.temperature,
#             max_tokens=self.max_tokens,
#             n=1,
#             response_format={"type": "json_object"}
#         )
#         return (resp.choices[0].message.content or "").strip()

#     def extract(self, note: str) -> List[Claim]:
#         # guard against extremely long pastes
#         safe_note = _truncate(note or "", _NOTE_MAX_CHARS)

#         # try once (+ optional light retry with a stricter reminder)
#         attempts = 1 + max(0, _RETRY)
#         raw = ""
#         last_err = None
#         for i in range(attempts):
#             try:
#                 raw = self._call_llm(safe_note if i == 0 else f"{safe_note}\n\nReturn ONLY the JSON object.")
#                 payload = _first_json_object(raw)
#                 data = json.loads(payload)
#                 claims = data.get("claims", [])
#                 if not isinstance(claims, list):
#                     claims = []
#                 break
#             except Exception as e:
#                 last_err = e
#                 logger.warning(f"ClaimExtractor attempt {i+1}/{attempts} failed to parse JSON: {e}")
#                 claims = []
#         else:
#             # exhausted attempts
#             logger.error(f"ClaimExtractor JSON parse failed after {attempts} attempt(s): {last_err}; raw={raw[:300]}")

#         # normalize & assign ids
#         out: List[Claim] = []
#         for c in claims:
#             c["id"] = c.get("id") or f"c_{uuid.uuid4().hex[:8]}"
#             c.setdefault("type", "efficacy")
#             c.setdefault("certainty", "may")
#             c.setdefault("pico", {"P": "", "I": "", "C": "", "O": ""})
#             # ensure required keys exist (defensive)
#             c["text"] = c.get("text", "").strip()
#             out.append(c)  # TypedDict is structural at runtime
#         return out

# agents/claim_extractor_openai.py
from __future__ import annotations
import json, logging, re, uuid
from typing import List, Dict
from openai import OpenAI
from .types import Claim

logger = logging.getLogger(__name__)
_client = OpenAI()

# Optional settings
try:
    from config.settings import settings
    _MODEL = getattr(settings, "OPENAI_CLAIM_MODEL", "gpt-4o-mini")
    _TEMP = getattr(settings, "TEMPERATURE_CLAIM", 0.0)
    _MAXTOK = getattr(settings, "CLAIM_MAX_TOKENS", 600)
    _RETRY = max(0, getattr(settings, "LLM_RETRY_COUNT", 1))
    _ENABLE_FALLBACK = getattr(settings, "ENABLE_CLAIM_FALLBACK", True)
except Exception:
    _MODEL, _TEMP, _MAXTOK, _RETRY, _ENABLE_FALLBACK = "gpt-4o-mini", 0.0, 600, 1, True

SYSTEM = (
    "You extract atomic medical claims from clinician notes. "
    "Return STRICT JSON with key 'claims' as a list of claim objects. "
    "Do NOT add prose."
)

USER_TMPL = """
Extract atomic, auditable medical claims.

Rules:
- Split multi-part sentences into separate claims.
- Preserve exact meaning; no new facts.
- Fill PICO when possible; leave empty strings if unknown.
- Classify type: efficacy, safety, dose, diagnostic, contraindication, marketing.
- Certainty from author wording: may, suggests, is proven, strongly, insufficient.

Return JSON ONLY:
{{"claims":[
  {{"id":"", "text":"", "type":"", "pico":{{"P":"","I":"","C":"","O":""}}, "certainty":""}}
]}}

Note:
{note}
""".strip()

# tiny heuristics for fallback
DOSE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ug|g|ml|mL|units|IU)\b", re.I)
EFF_WORDS = re.compile(r"\b(improves?|helps?|reduces?|relieves?|benefit[s]?)\b", re.I)
ABS_WORDS = re.compile(r"\b(cure[sd]?|guarantee[sd]?|always|never|proven)\b", re.I)
SAFE_WORDS = re.compile(r"\b(adverse|side effect|contraindicat|monitor|caution)\b", re.I)

def _new_id() -> str:
    return f"c_{uuid.uuid4().hex[:8]}"

def _normalize_claim(c: Dict) -> Claim:
    c["id"] = c.get("id") or _new_id()
    c.setdefault("text", "")
    c.setdefault("type", "efficacy")
    c.setdefault("certainty", "may")
    c.setdefault("pico", {"P": "", "I": "", "C": "", "O": ""})
    return c  # TypedDict relaxed at runtime

def _fallback_extract(note: str) -> List[Claim]:
    """Very small heuristic so the pipeline keeps moving if the LLM fails."""
    claims: List[Claim] = []
    # split by sentence-ish delimiters
    parts = [p.strip() for p in re.split(r"[.\n;]+", note or "") if p.strip()]
    for s in parts:
        t = "marketing"
        if DOSE_RE.search(s):
            t = "dose"
        elif SAFE_WORDS.search(s):
            t = "safety"
        elif EFF_WORDS.search(s) or ABS_WORDS.search(s):
            t = "efficacy"
        cert = "is proven" if ABS_WORDS.search(s) else "may"
        claims.append(_normalize_claim({"text": s, "type": t, "certainty": cert}))
    # de-dup by text
    seen, out = set(), []
    for c in claims:
        key = c["text"].lower()
        if key in seen: 
            continue
        seen.add(key)
        out.append(c)
    return out[:10]  # cap

class ClaimExtractor:
    def __init__(self, model: str = _MODEL, max_tokens: int = _MAXTOK, temperature: float = _TEMP):
        self.client = _client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _call_llm(self, note: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(note=note)},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,
            response_format={"type": "json_object"},  # ← force JSON
        )
        return (resp.choices[0].message.content or "").strip()

    def _parse_json(self, raw: str) -> List[Claim]:
        # if the model returned a bare list, wrap it
        txt = raw.strip()
        if txt.startswith("[") and txt.endswith("]"):
            txt = f'{{"claims": {txt}}}'
        # last-chance: pull the first JSON object
        m = re.search(r"\{.*\}", txt, flags=re.S)
        payload = m.group(0) if m else txt or "{}"
        data = json.loads(payload)
        claims = data.get("claims", [])
        if not isinstance(claims, list):
            return []
        return [_normalize_claim(dict(c)) for c in claims if isinstance(c, dict)]

    def extract(self, note: str) -> List[Claim]:
        attempts = 1 + max(0, _RETRY)
        last_err = None
        raw = ""
        for i in range(attempts):
            try:
                raw = self._call_llm(note)
                if raw:
                    claims = self._parse_json(raw)
                    if claims:
                        return claims
            except Exception as e:
                last_err = e
                logger.warning(f"ClaimExtractor attempt {i+1}/{attempts} failed to parse JSON: {e!s}")
        logger.error(f"ClaimExtractor JSON parse failed after {attempts} attempt(s): {last_err!s}; raw={raw[:200]}")
        # Heuristic fallback so the graph continues
        if _ENABLE_FALLBACK:
            fb = _fallback_extract(note or "")
            if fb:
                logger.info(f"ClaimExtractor fallback produced {len(fb)} claim(s).")
                return fb
        return []
