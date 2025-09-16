# # run_cli.py
# import argparse
# import sys
# from types import SimpleNamespace
# from pathlib import Path

# from document_processor.file_handler import DocumentProcessor
# from retriever.builder import RetrieverBuilder
# from agents.workflow import AgentWorkflow

# def _wrap_with_name(paths):
#     """Wrap plain file paths so DocumentProcessor can use .name."""
#     wrapped = []
#     for p in paths:
#         p = Path(p)
#         if not p.exists():
#             raise FileNotFoundError(f"File not found: {p}")
#         wrapped.append(SimpleNamespace(name=str(p)))
#     return wrapped

# def main():
#     parser = argparse.ArgumentParser(description="DocChat CLI runner (no Gradio)")
#     parser.add_argument("--question", "-q", required=True, help="Your question")
#     parser.add_argument("--files", "-f", required=True, nargs="+", help="One or more document paths")
#     parser.add_argument("--bm25-only", action="store_true", help="Skip vector store; use BM25 only")
#     args = parser.parse_args()

#     if args.bm25_only:
#     from langchain_community.retrievers import BM25Retriever
#     print("[CLI] Using BM25-only retriever")
#     retriever = BM25Retriever.from_documents(chunks)
# else:
#     retriever = retriever_builder.build_hybrid_retriever(chunks)

#     # 1) Prepare files and chunks
#     file_objs = _wrap_with_name(args.files)
#     processor = DocumentProcessor()
#     chunks = processor.process(file_objs)

#     if not chunks:
#         print("No chunks extracted from the provided files.")
#         sys.exit(1)

#     # 2) Build retriever
#     retriever_builder = RetrieverBuilder()
#     print("[CLI] building retriever...")
#     retriever = retriever_builder.build_hybrid_retriever(chunks)
#     print("[CLI] retriever built.")

#     print("[CLI] invoking workflow...")
#     result = workflow.full_pipeline(question=args.question, retriever=retriever)
#     print("[CLI] workflow done.")

#     #retriever = retriever_builder.build_hybrid_retriever(chunks)

#     # 3) Run workflow
#     workflow = AgentWorkflow()
#     result = workflow.full_pipeline(question=args.question, retriever=retriever)

#     # 4) Print results
#     print("\n=== DRAFT ANSWER ===\n")
#     print(result.get("draft_answer", ""))
#     print("\n=== VERIFICATION REPORT ===\n")
#     print(result.get("verification_report", ""))

# if __name__ == "__main__":
#     sys.exit(main())


# run_cli.py
import argparse
import sys
from types import SimpleNamespace
from pathlib import Path

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow


def _wrap_with_name(paths):
    """Wrap plain file paths so DocumentProcessor can use .name."""
    wrapped = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        wrapped.append(SimpleNamespace(name=str(p)))
    return wrapped


def main():
    parser = argparse.ArgumentParser(description="DocChat CLI runner (no Gradio)")
    parser.add_argument("--question", "-q", required=True, help="Your question")
    parser.add_argument(
        "--files", "-f", required=True, nargs="+", help="One or more document paths"
    )
    parser.add_argument(
        "--bm25-only", action="store_true", help="Skip vector store; use BM25 only"
    )
    args = parser.parse_args()

    # 1) Prepare files and chunks
    file_objs = _wrap_with_name(args.files)
    processor = DocumentProcessor()
    chunks = processor.process(file_objs)

    if not chunks:
        print("No chunks extracted from the provided files.")
        return 1

    # 2) Build retriever
    if args.bm25_only:
        print("[CLI] Using BM25-only retriever")
        try:
            from langchain_community.retrievers import BM25Retriever
        except Exception as e:
            print(f"Failed to import BM25Retriever: {e}")
            return 1
        retriever = BM25Retriever.from_documents(chunks)
    else:
        print("[CLI] Building hybrid retriever...")
        retriever_builder = RetrieverBuilder()
        retriever = retriever_builder.build_hybrid_retriever(chunks)
        print("[CLI] Retriever built.")

    # 3) Run workflow
    workflow = AgentWorkflow()
    print("[CLI] Invoking workflow...")
    result = workflow.full_pipeline(question=args.question, retriever=retriever)
    print("[CLI] Workflow done.")

    # 4) Print results
    print("\n=== DRAFT ANSWER ===\n")
    print(result.get("draft_answer", ""))

    print("\n=== VERIFICATION REPORT ===\n")
    print(result.get("verification_report", ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
