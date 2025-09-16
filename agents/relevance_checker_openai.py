# relevance_checker_openai.py
from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Any

from langchain.schema import Document
from openai import OpenAI

# Optional settings with safe fallbacks
try:
    from config.settings import settings
    _MODEL = getattr(settings, "OPENAI_CLAIM_MODEL", "gpt-4o-mini")   # or add OPENAI_SCOPE_MODEL in settings
    _TEMP = 0.0
    _MAXTOK = 10
    _RETRY = getattr(settings, "LLM_RETRY_COUNT", 1)
    _K_DEFAULT = 3
    _DOC_CHAR_LIMIT = 1200   # per doc snippet
    _TOTAL_CHAR_LIMIT = 6000 # overall cap
except Exception:
    _MODEL = "gpt-4o-mini"
    _TEMP = 0.0
    _MAXTOK = 10
    _RETRY = 1
    _K_DEFAULT = 3
    _DOC_CHAR_LIMIT = 1200
    _TOTAL_CHAR_LIMIT = 6000

logger = logging.getLogger(__name__)
_client = OpenAI()


class RelevanceChecker:
    def __init__(
        self,
        model: str = _MODEL,
        temperature: float = _TEMP,
        max_tokens: int = _MAXTOK,
    ):
        """
        Initialize the relevance checker using OpenAI Chat Completions.
        """
        self.client = _client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # --- helpers -------------------------------------------------------------

    def _retrieve(self, retriever: Any, query: str) -> List[Document]:
        """Work with either .invoke or .get_relevant_documents() across LC versions."""
        try:
            if hasattr(retriever, "invoke"):
                return retriever.invoke(query) or []
            if hasattr(retriever, "get_relevant_documents"):
                return retriever.get_relevant_documents(query) or []
        except Exception as e:
            logger.error(f"Retriever error: {e}")
        return []

    @staticmethod
    def _dedupe_docs(docs: List[Document]) -> List[Document]:
        """Dedupe by (doc_id,chunk_id) if present; else by content hash."""
        seen = set()
        out: List[Document] = []
        for d in docs:
            m = getattr(d, "metadata", {}) or {}
            key = (m.get("doc_id"), m.get("chunk_id"))
            if not key[0] or not key[1]:
                key = ("_", hashlib.sha256(d.page_content.encode("utf-8", errors="ignore")).hexdigest())
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out

    @staticmethod
    def _clip(text: str, n: int) -> str:
        if not text:
            return ""
        return text if len(text) <= n else text[:n]

    def _build_passages_block(self, docs: List[Document], k: int) -> str:
        """Join top-k doc snippets with per-doc and overall caps to keep prompt small."""
        pieces = []
        total = 0
        for i, d in enumerate(docs[:k], start=1):
            chunk = self._clip(d.page_content, _DOC_CHAR_LIMIT)
            if total + len(chunk) > _TOTAL_CHAR_LIMIT:
                break
            pieces.append(chunk)
            total += len(chunk)
        return "\n\n".join(pieces)

    # --- main ----------------------------------------------------------------

    def check(self, question: str, retriever: Any, k: int = _K_DEFAULT) -> str:
        """
        1) Retrieve top-k doc chunks via retriever
        2) Concatenate clipped text
        3) Ask the LLM to classify: CAN_ANSWER | PARTIAL | NO_MATCH
        """
        question = (question or "").strip()
        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")

        docs = self._retrieve(retriever, question)
        if not docs:
            logger.debug("No documents returned. Classifying as NO_MATCH.")
            return "NO_MATCH"

        docs = self._dedupe_docs(docs)
        document_content = self._build_passages_block(docs, k=max(1, k))

        user_prompt = f"""
You are an AI relevance checker between a user's question and provided document content.

Instructions:
- Classify how well the document content addresses the user's question.
- Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
- Do not include any additional text or explanation.

Labels:
1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all the details needed for a complete answer.
3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

Important: If the passages mention or reference the topic or timeframe of the question in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

Question: {question}

Passages:
{document_content}

Respond ONLY with one of: CAN_ANSWER, PARTIAL, NO_MATCH
""".strip()

        system_msg = (
            "You are a classifier that MUST reply with exactly one token from this set: "
            "CAN_ANSWER, PARTIAL, NO_MATCH. Do not add explanations."
        )

        # one call + optional light retry
        attempts = 1 + max(0, _RETRY)
        llm_response = ""
        last_err = None
        for i in range(attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt if i == 0 else (user_prompt + "\n\nReply with a single label only.")},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=1,
                )
                llm_response = (resp.choices[0].message.content or "").strip().upper()
                break
            except Exception as e:
                last_err = e
                logger.warning(f"RelevanceChecker attempt {i+1}/{attempts} failed: {e}")

        if not llm_response:
            logger.error(f"Model inference failed: {last_err}")
            return "NO_MATCH"

        if llm_response not in {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}:
            logger.debug(f"Invalid label '{llm_response}'. Forcing 'NO_MATCH'.")
            return "NO_MATCH"

        logger.debug(f"LLM response: {llm_response}")
        return llm_response
