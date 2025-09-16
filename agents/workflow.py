# agents/workflow.py
from __future__ import annotations

import operator
import re
from typing import Dict, List, Any
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
#rom langgraph.types import Send

# ✅ use the real module name
from .relevance_checker_openai import RelevanceChecker
from .claim_extractor_openai import ClaimExtractor
from .evidence_gatherer import EvidenceGatherer
from .evidence_normalizer import EvidenceNormalizer
from .verification_agent_openai import VerificationAgent
from .risk_detector import RiskyClaimDetector
from .safety_rewriter_openai import SafetyRewriter

from .types import Claim, Passage, Citation, Verdict, RiskFlag

try:
    from config.settings import settings
    _SKIP_SCOPE_GATE = getattr(settings, "SKIP_SCOPE_GATE", True)
except Exception:
    _SKIP_SCOPE_GATE = True


# # settings knobs (safe if settings is unavailable)
# try:
#     from config.settings import settings
#     _VERIFY_K = max(1, getattr(settings, "VERIFY_PASSAGES_PER_CLAIM", 3))
# except Exception:
#     _VERIFY_K = 3


# ============ Graph State ============
class AgentState(TypedDict):
    # Inputs
    question: str
    clinician_note: str
    retriever: Any

    # Scope gate
    is_relevant: bool

    # Claim-centric flow
    claims: List[Claim]
    claim_passages: Dict[str, List[Passage]]
    citations: List[Citation]

    # Map step accumulates verdicts with a reducer (append lists)
    verdicts: Annotated[List[Verdict], operator.add]

    # Downstream
    risk_flags: List[RiskFlag]
    rewritten_note: str

    # Compatibility with your UI
    draft_answer: str
    verification_report: str


# ============ Workflow ===============
# class AgentWorkflow:
#     def __init__(self):
#         self.scope = RelevanceChecker()
#         self.extractor = ClaimExtractor()
#         # let EvidenceGatherer read k from settings internally
#         self.gatherer = EvidenceGatherer()
#         self.normalizer = EvidenceNormalizer()
#         self.verifier = VerificationAgent()
#         self.risk = RiskyClaimDetector()
#         self.rewriter = SafetyRewriter()
#         self.compiled_workflow = self._build()

#     def _build(self):
#         g = StateGraph(AgentState)

#         # Nodes
#         g.add_node("check_relevance", self._check_relevance)
#         g.add_node("extract_claims", self._extract_claims)
#         g.add_node("gather_evidence", self._gather_evidence)
#         g.add_node("normalize_evidence", self._normalize_evidence)
#         g.add_node("plan_verification", self._plan_verification)  # fan-out planner
#         g.add_node("verify_one", self._verify_one)                # map worker
#         g.add_node("detect_risks", self._detect_risks)            # fan-in continues here
#         g.add_node("rewrite_note", self._rewrite_note)
#         g.add_node("summarize_report", self._summarize_report)

#         # Edges
#         g.add_edge(START, "check_relevance")
#         g.add_conditional_edges("check_relevance", self._after_scope,
#                                 {"relevant": "extract_claims", "irrelevant": END})
#         g.add_edge("extract_claims", "gather_evidence")
#         g.add_edge("gather_evidence", "normalize_evidence")
#         g.add_edge("normalize_evidence", "plan_verification")

#         # Map pattern: planner → (many verify_one) … then continue from the planner
#         # ❗ Do NOT add an edge from verify_one → next; fan-in happens via the planner.
#         g.add_edge("plan_verification", "detect_risks")

#         g.add_edge("detect_risks", "rewrite_note")
#         g.add_edge("rewrite_note", "summarize_report")
#         g.add_edge("summarize_report", END)

#         return g.compile()

class AgentWorkflow:
    def __init__(self):
        self.scope = RelevanceChecker()
        self.extractor = ClaimExtractor()
        self.gatherer = EvidenceGatherer()
        self.normalizer = EvidenceNormalizer()
        self.verifier = VerificationAgent()
        self.risk = RiskyClaimDetector()
        self.rewriter = SafetyRewriter()
        self.compiled_workflow = self._build()

    def _build(self):
        g = StateGraph(AgentState)

        # Nodes
        g.add_node("check_relevance", self._check_relevance)
        g.add_node("extract_claims", self._extract_claims)
        g.add_node("gather_evidence", self._gather_evidence)
        g.add_node("normalize_evidence", self._normalize_evidence)

        # ⬇️ single node that verifies all claims sequentially
        g.add_node("verify_claims", self._verify_claims)

        g.add_node("detect_risks", self._detect_risks)
        g.add_node("rewrite_note", self._rewrite_note)
        g.add_node("summarize_report", self._summarize_report)

        # Edges
        g.add_edge(START, "check_relevance")
        g.add_conditional_edges("check_relevance", self._after_scope,
                                {"relevant": "extract_claims", "irrelevant": END})
        g.add_edge("extract_claims", "gather_evidence")
        g.add_edge("gather_evidence", "normalize_evidence")

        # ⬇️ no fan-out / fan-in, just go to verification
        g.add_edge("normalize_evidence", "verify_claims")
        g.add_edge("verify_claims", "detect_risks")

        g.add_edge("detect_risks", "rewrite_note")
        g.add_edge("rewrite_note", "summarize_report")
        g.add_edge("summarize_report", END)

        return g.compile()

    
        # NEW: verify all claims in one node
    def _verify_claims(self, s: AgentState) -> Dict:
        claims = s.get("claims", []) or []
        if not claims:
            return {"verdicts": []}
        verdicts = self.verifier.verify(
            claims=claims,
            claim_passages=s.get("claim_passages", {}),
            citations=s.get("citations", []),
            k_passages=max(1, getattr(settings, "VERIFY_PASSAGES_PER_CLAIM", 3)),
        )
        return {"verdicts": verdicts}

    # ---------- public API ----------
    def full_pipeline(self, question: str, retriever: Any, clinician_note: str = ""):
        initial: AgentState = {
            "question": question,
            "clinician_note": clinician_note or question,
            "retriever": retriever,
            "is_relevant": False,

            "claims": [],
            "claim_passages": {},
            "citations": [],

            "verdicts": [],  # reducer will append
            "risk_flags": [],
            "rewritten_note": "",

            "draft_answer": "",
            "verification_report": "",
        }
        # ... inside AgentWorkflow.full_pipeline(...)
        final = self.compiled_workflow.invoke(initial)
        return {
            "rewritten_note": final.get("rewritten_note", ""),
            "draft_answer": final.get("rewritten_note") or final.get("draft_answer"),
            "verification_report": final.get("verification_report", ""),
            # ⬇️ new: pass through for the UI
            "claims": final.get("claims", []),
            "verdicts": final.get("verdicts", []),
            "risk_flags": final.get("risk_flags", []),
        }

        # final = self.compiled_workflow.invoke(initial)
        # return {
        #     "rewritten_note": final.get("rewritten_note", ""),
        #     "draft_answer": final.get("rewritten_note") or final.get("draft_answer"),
        #     "verification_report": final.get("verification_report", ""),
        # }

    # ---------- nodes ----------
    # def _check_relevance(self, s: AgentState) -> Dict:
    #     label = self.scope.check(question=s["question"], retriever=s["retriever"], k=10)
    #     return {"is_relevant": label in ("CAN_ANSWER", "PARTIAL")}

    def _check_relevance(self, s: AgentState) -> Dict:
    # If we have a clinician note (rewrite workflow), skip scope gate by default
     if _SKIP_SCOPE_GATE and (s.get("clinician_note") or "").strip(): return {"is_relevant": True}
     label = self.scope.check(question=s["question"], retriever=s["retriever"], k=10)
     return {"is_relevant": label in ("CAN_ANSWER", "PARTIAL")}

    def _after_scope(self, s: AgentState) -> str:
        return "relevant" if s["is_relevant"] else "irrelevant"

    def _extract_claims(self, s: AgentState) -> Dict:
        claims = self.extractor.extract(s["clinician_note"])
        summary = f"Extracted {len(claims)} claim(s)."
        return {"claims": claims, "draft_answer": summary}

    def _gather_evidence(self, s: AgentState) -> Dict:
        cp = self.gatherer.gather(s["claims"], s["retriever"])
        return {"claim_passages": cp}

    def _normalize_evidence(self, s: AgentState) -> Dict:
        citations = self.normalizer.normalize(s["claim_passages"])
        return {"citations": citations}

    # ---- MAP: fan-out to verify_one (one Send per claim) ----
    # def _plan_verification(self, s: AgentState):
    #     sends: List[Send] = []
    #     for c in s["claims"]:
    #         passages = s["claim_passages"].get(c["id"], [])
    #         claim_citations = [ct for ct in s["citations"] if ct.get("claim_id") == c["id"]]
    #         sends.append(Send("verify_one", {"claim": c, "passages": passages, "citations": claim_citations}))
    #     return sends  # after all complete, graph continues from this planner node
    def _plan_verification(self, s: AgentState):
        sends: List[Send] = []
        for c in s["claims"]:
            passages = s["claim_passages"].get(c["id"], [])
            claim_citations = [ct for ct in s["citations"] if ct.get("claim_id") == c["id"]]
            sends.append(Send("verify_one", {"claim": c, "passages": passages, "citations": claim_citations}))

        # ✅ Important: when no items, return a dict (not [])
        if not sends:
            return {"verdicts": []}

        return sends


    # ---- MAP worker: verify a single claim, append to verdicts via reducer ----
    def _verify_one(self, s: AgentState) -> Dict:
        claim: Claim = s["claim"]      # injected by Send(...)
        passages: List[Passage] = s.get("passages", [])
        citations: List[Citation] = s.get("citations", [])

        v = self.verifier.verify(
            claims=[claim],
            claim_passages={claim["id"]: passages},
            citations=citations,
            k_passages=_VERIFY_K,
        )[0]
        return {"verdicts": [v]}

    def _detect_risks(self, s: AgentState) -> Dict:
        flags = self.risk.detect(
            clinician_note=s.get("clinician_note", ""),
            claims=s["claims"],
            verdicts=s.get("verdicts", []),
        )
        return {"risk_flags": flags}

    def _rewrite_note(self, s: AgentState) -> Dict:
        rewritten = self.rewriter.rewrite(
            note=s.get("clinician_note", ""),
            claims=s["claims"],
            verdicts=s.get("verdicts", []),
            citations=s.get("citations", []),
            risks=s.get("risk_flags", []),
        )
        return {"rewritten_note": rewritten}

    def _summarize_report(self, s: AgentState) -> Dict:
        summary = self.verifier.summarize(s.get("verdicts", []))
        return {"verification_report": summary}
