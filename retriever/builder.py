# from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
# from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
# from langchain_ibm import WatsonxEmbeddings
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# from config.settings import settings
# import logging

# logger = logging.getLogger(__name__)

# class RetrieverBuilder:
#     def __init__(self):
#         """Initialize the retriever builder with embeddings."""
#         embed_params = {
#             EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
#             EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
#         }

#         watsonx_embedding = WatsonxEmbeddings(
#             model_id="ibm/slate-125m-english-rtrvr",
#             url="https://us-south.ml.cloud.ibm.com",
#             project_id="skills-network",
#             params=embed_params
#         )
#         self.embeddings = watsonx_embedding
        
#     def build_hybrid_retriever(self, docs):
#         """Build a hybrid retriever using BM25 and vector-based retrieval."""
#         try:
#             # Create Chroma vector store
#             vector_store = Chroma.from_documents(
#                 documents=docs,
#                 embedding=self.embeddings,
#                 persist_directory=settings.CHROMA_DB_PATH
#             )
#             logger.info("Vector store created successfully.")
            
#             # Create BM25 retriever
#             bm25 = BM25Retriever.from_documents(docs)
#             logger.info("BM25 retriever created successfully.")
            
#             # Create vector-based retriever
#             vector_retriever = vector_store.as_retriever(search_kwargs={"k": settings.VECTOR_SEARCH_K})
#             logger.info("Vector retriever created successfully.")
            
#             # Combine retrievers into a hybrid retriever
#             hybrid_retriever = EnsembleRetriever(
#                 retrievers=[bm25, vector_retriever],
#                 weights=settings.HYBRID_RETRIEVER_WEIGHTS
#             )
#             logger.info("Hybrid retriever created successfully.")
#             return hybrid_retriever
#         except Exception as e:
#             logger.error(f"Failed to build hybrid retriever: {e}")
#             raise

# retriever/builder.py
from pathlib import Path
import logging
from typing import List, Optional
import shutil
import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

from config.settings import settings

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self):
        """
        Initialize the retriever builder with OpenAI embeddings.
        """
        self.model = getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.embeddings = OpenAIEmbeddings(model=self.model)

        # Root where per-session Chroma dirs will live
        self.persist_root = Path(settings.CHROMA_DB_PATH) / self.model.replace("/", "_")
        self.persist_root.mkdir(parents=True, exist_ok=True)

    def _session_paths(self, session_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """
        Return (persist_directory, collection_name) based on session_id.
        If session_id is None -> in-memory (no persistence, zero leakage).
        """
        if not session_id:
            return None, None
        persist_dir = str(self.persist_root / session_id)
        collection_name = f"{settings.CHROMA_COLLECTION_NAME}_{session_id}"
        return persist_dir, collection_name

    def _load_or_create_chroma(
        self,
        docs: List[Document],
        persist_dir: Optional[str],
        collection_name: Optional[str],
    ) -> Chroma:
        """
        If a persisted collection exists, load it; otherwise create from documents.
        """
        if persist_dir and os.path.isdir(persist_dir):
            try:
                vs = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_dir,
                )
                # If empty (e.g., folder exists but no data), fall through to create.
                # Chroma exposes _collection.count() on the underlying client.
                if getattr(vs, "_collection", None):
                    count = vs._collection.count()  # type: ignore[attr-defined]
                    if count and count > 0:
                        logger.info(f"Loaded existing Chroma collection '{collection_name}' ({count} vectors).")
                        return vs
            except Exception as e:
                logger.warning(f"Could not load existing Chroma at {persist_dir}: {e}. Recreating.")

        # Create a fresh collection from docs
        vs = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=persist_dir,     # None => in-memory
            collection_name=collection_name,   # None => default in-memory name
            collection_metadata={"session_id": collection_name or "in_memory",
                                 "embedding_model": self.model},
        )
        if persist_dir:
            try:
                vs.persist()
            except Exception as e:
                logger.warning(f"Persist call failed for {persist_dir}: {e}")
        logger.info(
            f"Vector store {'created' if persist_dir else 'created in-memory'}"
            + (f" at {persist_dir} (collection='{collection_name}')." if persist_dir else ".")
        )
        return vs

    def build_hybrid_retriever(self, docs: List[Document], session_id: Optional[str] = None):
        """
        Build a hybrid retriever (BM25 + Vector).
        Set a session_id (e.g., hash of uploaded files) to isolate collections between runs.
        """
        if not docs:
            raise ValueError("No documents provided to build the retriever.")

        try:
            persist_dir, collection_name = self._session_paths(session_id)

            vector_store = self._load_or_create_chroma(
                docs=docs,
                persist_dir=persist_dir,
                collection_name=collection_name,
            )

            # BM25 retriever (consistent k)
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = settings.VECTOR_SEARCH_K
            logger.info(f"BM25 retriever created (k={bm25.k}).")

            # Vector retriever (k with generous fetch_k for recall)
            vector_retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": settings.VECTOR_SEARCH_K,
                    "fetch_k": settings.VECTOR_SEARCH_K * 4
                }
            )
            logger.info(f"Vector retriever created (k={settings.VECTOR_SEARCH_K}).")

            # Hybrid (BM25 + Vector)
            hybrid = EnsembleRetriever(
                retrievers=[bm25, vector_retriever],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS,
            )
            logger.info("Hybrid retriever created successfully.")
            return hybrid

        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise

    def drop_session(self, session_id: str) -> None:
        """
        Optional utility: delete a session's persisted Chroma directory (cleanup).
        """
        persist_dir, _ = self._session_paths(session_id)
        if persist_dir and Path(persist_dir).exists():
            shutil.rmtree(persist_dir, ignore_errors=True)
            logger.info(f"Dropped Chroma session at {persist_dir}")
