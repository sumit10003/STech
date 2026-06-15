import os

# Prevent HuggingFace from attempting network checks (recommended for offline use)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HUGGING_FACE_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src import config

class EmbeddingPipeline:
    def __init__(self, model_name: str = None, chunk_size: int = 500, chunk_overlap: int = 100):
        """Initialize embedding pipeline.

        If `model_name` is None, reads default from `src.config.EMBEDDING_MODEL`.
        The `model_name` may be either a HuggingFace model identifier or a local
        path to a downloaded model folder (recommended for offline use).
        """
        if model_name is None:
            model_name = getattr(config, "EMBEDDING_MODEL", "all-mpnet-base-v2")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Force local-only model resolution in offline deployments.
        self.model = SentenceTransformer(model_name, local_files_only=True)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents(config.ADMIN_DATA_DIR)
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
