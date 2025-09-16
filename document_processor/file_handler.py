# document_processor/file_handler.py
import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import constants
from config.settings import settings
from utils.logging import logger

# Safe fallbacks if settings doesn't define these
_CHUNK_SIZE = getattr(settings, "CHUNK_SIZE", 1200)
_CHUNK_OVERLAP = getattr(settings, "CHUNK_OVERLAP", 150)

class DocumentProcessor:
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def validate_files(self, files: List) -> None:
        # Enforce allowed extensions & total size
        for f in files:
            ext = os.path.splitext(f.name)[1].lower()
            if ext not in constants.ALLOWED_TYPES:
                raise ValueError(f"Unsupported file type: {ext}. Allowed: {constants.ALLOWED_TYPES}")
        total_size = sum(os.path.getsize(f.name) for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE//1024//1024}MB limit")

    def process(self, files: List) -> List[Document]:
        self.validate_files(files)
        all_chunks: List[Document] = []
        seen_hashes = set()

        for file in files:
            try:
                # stream hash (memory friendly for large files)
                h = hashlib.sha256()
                with open(file.name, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                file_hash = h.hexdigest()

                cache_path = self.cache_dir / f"{file_hash}.pkl"
                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {file.name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"Processing and caching: {file.name}")
                    chunks = self._process_file(file, file_hash=file_hash)
                    self._save_to_cache(chunks, cache_path)

                # global dedupe by content hash (after metadata attached)
                for chunk in chunks:
                    chunk_hash = hashlib.sha256(chunk.page_content.encode("utf-8", errors="ignore")).hexdigest()
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                logger.error(f"Failed to process {file.name}: {str(e)}")
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    def _process_file(self, file, file_hash: str) -> List[Document]:
        """
        Process a single file. Use Docling for PDFs/DOCX, direct reads for TXT/MD.
        Attaches rich metadata to each chunk so downstream agents can cite sources.
        """
        name = file.name
        ext = os.path.splitext(name)[1].lower()
        abs_path = os.path.abspath(name)
        doc_id = file_hash[:12]
        base_meta = {
            "source": name,
            "file_path": abs_path,
            "doc_id": doc_id,
            "ext": ext,
        }

        if ext not in (".pdf", ".docx", ".txt", ".md"):
            logger.warning(f"Skipping unsupported file type: {name}")
            return []

        # ----- helpers -----
        def rc_split_documents(parents: List[Document], prefix: str) -> List[Document]:
            rc = RecursiveCharacterTextSplitter(
                chunk_size=_CHUNK_SIZE,
                chunk_overlap=_CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = rc.split_documents(parents)
            for j, c in enumerate(chunks):
                parent = c.metadata.get("chunk_parent") or f"{doc_id}-{prefix}"
                c.metadata.setdefault("chunk_id", f"{parent}-{j:03d}")
            # clean helper field
            for c in chunks:
                c.metadata.pop("chunk_parent", None)
            return chunks

        # ----- TXT -----
        if ext == ".txt":
            try:
                with open(name, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except Exception as e:
                logger.error(f"Failed to read text file {name}: {e}")
                return []
            if not text.strip():
                return []
            parent = Document(page_content=text, metadata=base_meta | {"chunk_parent": f"{doc_id}-t"})
            return rc_split_documents([parent], prefix="t")

        # ----- MD -----
        if ext == ".md":
            try:
                with open(name, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except Exception as e:
                logger.error(f"Failed to read markdown file {name}: {e}")
                return []
            if not text.strip():
                return []
            # First: header-aware split → parent docs with header metadata
            header_splitter = MarkdownHeaderTextSplitter(self.headers)
            header_docs = header_splitter.split_text(text)
            parents: List[Document] = []
            for i, d in enumerate(header_docs):
                md = dict(d.metadata or {})
                md.update(base_meta)
                md["chunk_parent"] = f"{doc_id}-m-{i:04d}"
                parents.append(Document(page_content=d.page_content, metadata=md))
            # Then: recursive split so chunks are reasonably sized
            return rc_split_documents(parents, prefix="m")

        # ----- PDF / DOCX via Docling -----
        try:
            converter = DocumentConverter()
            result = converter.convert(name)
            markdown = result.document.export_to_markdown()
        except Exception as e:
            logger.error(f"Docling conversion failed for {name}: {e}")
            return []

        header_splitter = MarkdownHeaderTextSplitter(self.headers)
        header_docs = header_splitter.split_text(markdown)
        parents: List[Document] = []
        for i, d in enumerate(header_docs):
            md = dict(d.metadata or {})
            md.update(base_meta)
            md["chunk_parent"] = f"{doc_id}-p-{i:04d}"
            parents.append(Document(page_content=d.page_content, metadata=md))

        return rc_split_documents(parents, prefix="p")

    def _save_to_cache(self, chunks: List[Document], cache_path: Path):
        with open(cache_path, "wb") as f:
            pickle.dump({"timestamp": datetime.now().timestamp(), "chunks": chunks}, f)

    def _load_from_cache(self, cache_path: Path) -> List[Document]:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)
