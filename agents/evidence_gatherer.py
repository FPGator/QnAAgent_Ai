# agents/evidence_gatherer.py
from __future__ import annotations

import hashlib
import logging
from typing import List, Dict, Any, Optional

from langchain.schema import Document
from .types import Claim, Passage

logger = logging.getLogger(__name__)

def _mk_query(claim: Claim) -> str:
    """
    Enrich the retrieval query with PICO fields when present.
    This typically improves retrieval for clinical text.
    """
    pico = claim.get("pico") or {}
    P = pico.get("P", "")
    I = pico.get("I", "")
    C = pico.get("C", "")
    O = pico.get("O", "")
    parts = [claim.get("text", "").strip()]
    context_bits = []
    if P: context_bits.append(f"P:{P}")
    if I: context_bits.append(f"I:{I}")
    if C: context_bits.append(f"C:{C}")
    if O: context_bits.append(f"O:{O}")
    if context_bits:
        parts.append(" | ".join(context_bits))
    return " ".join(parts).strip()


class EvidenceGatherer:
    def __init__(self, k: Optional[int] = None):
        # Pull default from settings if available
        try:
            from config.settings import settings
            self.k = k or settings.RETRIEVAL_TOPK_PER_CLAIM
        except Exception:
            self.k = k or 4

    def _retrieve(self, retriever: Any, query: str) -> List[Document]:
        """
        Call retriever robustly across LangChain versions:
        - Prefer .invoke(query)
        - Fallback to .get_relevant_documents(query)
        """
        docs: List[Document] = []
        try:
            if hasattr(retriever, "invoke"):
                docs = retriever.invoke(query) or []
            elif hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(query) or []
            else:
                raise AttributeError("Retriever has neither .invoke nor .get_relevant_documents")
        except Exception as e:
            logger.error(f"Retriever error on query='{query[:120]}...': {e}")
            return []
        return docs

    def _dedupe_docs(self, docs: List[Document]) -> List[Document]:
        """
        Deduplicate by (doc_id, chunk_id) when present; fallback to content hash.
        """
        seen = set()
        out: List[Document] = []
        for d in docs:
            m = getattr(d, "metadata", {}) or {}
            key = (m.get("doc_id"), m.get("chunk_id"))
            if not key[0] or not key[1]:
                # fallback content hash to avoid dupes from different retriever legs
                key = ("_", hashlib.sha256(d.page_content.encode("utf-8", errors="ignore")).hexdigest())
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out

    def gather(self, claims: List[Claim], retriever: Any) -> Dict[str, List[Passage]]:
        """
        For each claim, pull top-k chunks from the retriever.
        Returns: claim_id -> list[Passage]
        """
        claim_passages: Dict[str, List[Passage]] = {}

        for claim in claims:
            query = _mk_query(claim)
            docs = self._retrieve(retriever, query)
            docs = self._dedupe_docs(docs)[: self.k]

            passages: List[Passage] = []
            for d in docs:
                meta = getattr(d, "metadata", {}) or {}
                src = meta.get("source") or meta.get("file_path") or meta.get("url")
                # If similarity score exists (varies by backend), keep it in metadata
                if "score" not in meta and meta.get("distance") is not None:
                    # normalize a bit: lower distance → higher score
                    try:
                        meta["score"] = 1.0 / (1.0 + float(meta["distance"]))
                    except Exception:
                        pass

                passages.append(Passage(
                    claim_id=claim["id"],
                    text=d.page_content,
                    source=src,
                    metadata=meta
                ))

            claim_passages[claim["id"]] = passages

        return claim_passages

    @staticmethod
    def flatten_passages(claim_passages: Dict[str, List[Passage]]) -> List[Document]:
        """
        If you need a single list[Document] (legacy paths), rebuild Documents from Passages.
        """
        out: List[Document] = []
        for plist in claim_passages.values():
            for p in plist:
                out.append(Document(page_content=p["text"], metadata=p.get("metadata") or {}))
        return out
