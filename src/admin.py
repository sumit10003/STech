import os
from pathlib import Path
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

from src import config

class AdminManager:
    """Admin interface for managing documents and vector store"""
    
    def __init__(self, data_dir: str = None, vector_store_dir: str = None):
        # Use config defaults if not specified
        if data_dir is None:
            data_dir = config.ADMIN_DATA_DIR
        if vector_store_dir is None:
            vector_store_dir = config.VECTOR_STORE_DIR
        self.data_dir = data_dir
        self.vector_store_dir = vector_store_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.vector_store_dir, exist_ok=True)
    
    def get_supported_formats(self) -> list:
        """Return list of supported file formats"""
        return ["PDF", "TXT", "CSV", "XLSX", "DOCX", "JSON"]
    
    def get_uploaded_files(self) -> list:
        """Get list of files currently in admin_data folder"""
        data_path = Path(self.data_dir)
        if not data_path.exists():
            return []
        
        files = []
        for ext in ['*.pdf', '*.txt', '*.csv', '*.xlsx', '*.docx', '*.json']:
            files.extend([{"name": f.name, "type": ext.replace('*', ''), "size": f.stat().st_size} 
                         for f in data_path.glob(ext)])
        
        return files
    
    def build_vector_store(self, progress_callback=None) -> dict:
        """
        Load documents from admin_data and build vector store.
        
        Args:
            progress_callback: Optional callback function(step, total_steps, message)
                              to report progress during build
        
        Returns status dict with success/error info
        """
        def report_progress(step, total, msg):
            if progress_callback:
                progress_callback(step, total, msg)
        
        try:
            total_steps = 5
            
            # Step 1: Load documents
            report_progress(1, total_steps, "📂 Loading documents...")
            docs = load_all_documents(self.data_dir)
            
            if not docs:
                return {
                    "success": False,
                    "message": "No documents found in admin_data folder",
                    "doc_count": 0
                }
            
            report_progress(2, total_steps, f"✅ Loaded {len(docs)} document pages")
            
            # Step 3: Build vector store (includes chunking + embedding)
            report_progress(3, total_steps, "🔄 Chunking documents...")
            store = FaissVectorStore(self.vector_store_dir)
            
            report_progress(4, total_steps, "🧠 Generating embeddings (this takes time)...")
            store.build_from_documents(docs)
            
            report_progress(5, total_steps, "💾 Saving vector store...")
            
            return {
                "success": True,
                "message": f"Vector store built successfully with {len(docs)} documents",
                "doc_count": len(docs)
            }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error building vector store: {str(e)}",
                "doc_count": 0
            }
    
    def check_vector_store_status(self) -> dict:
        """Check if vector store exists and is valid"""
        faiss_path = os.path.join(self.vector_store_dir, "faiss.index")
        meta_path = os.path.join(self.vector_store_dir, "metadata.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            return {
                "exists": True,
                "faiss_path": faiss_path,
                "meta_path": meta_path
            }
        else:
            return {
                "exists": False,
                "message": "Vector store not yet created"
            }
