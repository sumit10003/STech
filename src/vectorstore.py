import os

# Prevent HuggingFace from attempting network checks (recommended for offline use)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HUGGING_FACE_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import faiss
import numpy as np
import pickle
import json
from typing import List, Any, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline
from src import config

class FaissVectorStore:
    def __init__(self, persist_dir: str = None, embedding_model: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Faiss-backed vector store with enhanced metadata management.

        `embedding_model` may be a HF id or a local path. If None, falls back to
        `src.config.EMBEDDING_MODEL` to allow offline use.
        """
        # Use config defaults if not specified
        if persist_dir is None:
            persist_dir = config.VECTOR_STORE_DIR
        if embedding_model is None:
            embedding_model = config.EMBEDDING_MODEL
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.document_metadata = {}  # Track document-level metadata
        self.embedding_model = embedding_model
        # Use device='cpu' to avoid meta tensor issues with torch
        self.model = SentenceTransformer(embedding_model, device='cpu')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        """Build vector store from documents with enhanced metadata tracking"""
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        
        # Track document sources
        doc_sources = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown") if hasattr(doc, "metadata") else "unknown"
            if source not in doc_sources:
                doc_sources[source] = {
                    "source": source,
                    "page_count": 0,
                    "added_at": datetime.now().isoformat(),
                    "status": "processed"
                }
            doc_sources[source]["page_count"] += 1
        
        self.document_metadata = doc_sources
        
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        
        # Create enhanced metadata with source tracking and equipment name
        from src.data_loader import extract_equipment_name
        metadatas = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown") if hasattr(chunk, "metadata") else "unknown"
            equipment_name = extract_equipment_name(source)
            metadatas.append({
                "text": chunk.page_content,
                "source": source,
                "page": chunk.metadata.get("page", 0) if hasattr(chunk, "metadata") else 0,
                "equipment_name": equipment_name,
                "timestamp": datetime.now().isoformat()
            })
        
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built with {len(chunks)} chunks from {len(doc_sources)} documents")
        print(f"[INFO] Vector store saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        """Save FAISS index, metadata, and document metadata"""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        doc_meta_path = os.path.join(self.persist_dir, "documents_metadata.json")
        
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        # Save document metadata as JSON for easy inspection
        with open(doc_meta_path, "w") as f:
            json.dump(self.document_metadata, f, indent=2)
        
        print(f"[INFO] Saved Faiss index, metadata, and document info to {self.persist_dir}")

    def load(self):
        """Load FAISS index, metadata, and document metadata"""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        doc_meta_path = os.path.join(self.persist_dir, "documents_metadata.json")
        
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        # Load document metadata if available
        if os.path.exists(doc_meta_path):
            with open(doc_meta_path, "r") as f:
                self.document_metadata = json.load(f)
        
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")
    
    def get_document_sources(self) -> Dict[str, Any]:
        """Get all document sources and their metadata"""
        return self.document_metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_vectors": len(self.metadata) if self.metadata else 0,
            "total_documents": len(self.document_metadata),
            "documents": self.document_metadata,
            "embedding_model": self.embedding_model,
            "index_dimension": self.index.d if self.index else 0
        }

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5, equipment_filter: str = None):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        if equipment_filter:
            print(f"[INFO] Filtering by equipment: {equipment_filter}")
        query_emb = self.model.encode([query_text]).astype('float32')
        
        # If equipment filter specified, get more results and filter
        if equipment_filter:
            # Get 3x more results to ensure we have enough after filtering
            results = self.search(query_emb, top_k=top_k * 5)
            # Filter by equipment name
            filtered_results = [
                r for r in results 
                if r.get('metadata', {}).get('equipment_name', '').lower() == equipment_filter.lower()
            ]
            print(f"[INFO] Found {len(filtered_results)} results for equipment '{equipment_filter}'")
            return filtered_results[:top_k]
        else:
            return self.search(query_emb, top_k=top_k)
    
    def get_available_equipment(self) -> List[str]:
        """Get list of unique equipment names in the vector store"""
        if not self.metadata:
            return []
        equipment_set = set()
        for meta in self.metadata:
            equip_name = meta.get('equipment_name', '')
            if equip_name and equip_name != 'unknown':
                equipment_set.add(equip_name)
        return sorted(list(equipment_set))
