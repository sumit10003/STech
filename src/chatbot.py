import os
from typing import Dict, Generator, List, Optional
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src import config

load_dotenv()

class RAGChatbot:
    """RAG-based chatbot for user interactions"""
    
    def __init__(self, persist_dir: str = None, embedding_model: Optional[str] = None, llm_model: Optional[str] = None):
        if persist_dir is None:
            persist_dir = config.VECTOR_STORE_DIR
        if embedding_model is None:
            embedding_model = config.EMBEDDING_MODEL
        if llm_model is None:
            llm_model = config.LLM_MODEL

        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        
        # Load vectorstore if it exists
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.vectorstore.load()
            print(f"[INFO] Loaded existing vector store from {persist_dir}")
        else:
            print(f"[WARN] Vector store not found at {persist_dir}. Run admin panel to create one.")
            self.vectorstore = None
        
        # Initialize LLM with config settings for factual responses
        # Temperature 0 = completely factual, no creativity (best for RAG/technical docs)
        self.llm = ChatOllama(
            model=llm_model,
            temperature=config.LLM_TEMPERATURE,  # 0 = factual, no hallucination
            num_predict=config.LLM_MAX_TOKENS,  # Max tokens in response
            top_p=config.LLM_TOP_P,  # Nucleus sampling
        )
        print(f"[INFO] ChatOllama LLM initialized: {llm_model} (temperature={config.LLM_TEMPERATURE}, factual mode)")

    def _prepare_prompt_payload(
        self,
        query: str,
        top_k: int = 3,
        equipment_name: Optional[str] = None,
    ) -> Dict:
        """Prepare retrieval context and prompt for both sync and streaming responses."""
        # Validate query length
        if len(query) > config.QUERY_CHAR_LIMIT:
            return {
                "error": {
                    "answer": (
                        f"Query exceeds maximum character limit of {config.QUERY_CHAR_LIMIT} "
                        f"characters. Current length: {len(query)} characters."
                    ),
                    "sources": [],
                    "follow_up_questions": [],
                }
            }

        if self.vectorstore is None:
            return {
                "error": {
                    "answer": "Vector store not initialized. Please contact admin to load documents.",
                    "sources": [],
                    "follow_up_questions": [],
                }
            }

        # Retrieve relevant documents (with equipment filtering if specified)
        results = self.vectorstore.query(query, top_k=top_k, equipment_filter=equipment_name)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]

        if not texts:
            if equipment_name:
                return {
                    "error": {
                        "answer": (
                            f"No relevant information found for equipment '{equipment_name}' in the "
                            "knowledge base. Please verify the equipment name or try a different query."
                        ),
                        "sources": [],
                        "follow_up_questions": [],
                    }
                }
            return {
                "error": {
                    "answer": "No relevant information found in the knowledge base.",
                    "sources": [],
                    "follow_up_questions": [],
                }
            }

        # Build detailed context with metadata (required for citations)
        context_parts = []
        for r in results:
            metadata = r.get("metadata", {})
            source_path = metadata.get("source", "Unknown Document")

            # Extract filename from full path and normalize extensions for cleaner citations
            if "\\" in source_path or "/" in source_path:
                source_name = source_path.replace("\\", "/").split("/")[-1]
                source_name = source_name.replace(".pdf", "").replace(".txt", "").replace(".docx", "")
            else:
                source_name = source_path

            page_num = metadata.get("page", "N/A")
            page_label = metadata.get("page_label", page_num)
            text_content = metadata.get("text", "")
            context_parts.append(f"SOURCE: {source_name} (Page {page_label})\nCONTENT: {text_content}")

        context = "\n\n---\n\n".join(context_parts)
        equipment_context = (
            f"\n### EQUIPMENT CONTEXT\nAll information below is specific to: **{equipment_name}**\n"
            if equipment_name
            else ""
        )

        prompt = f"""
        ### ROLE
        You are a Senior MRO (Maintenance, Repair, and Operations) Technical Expert. Your goal is to provide highly detailed, accurate, and cited technical guidance.{equipment_context}

        ### CONTEXT DOCUMENTS
        {context}

        ### INSTRUCTIONS
        - **Detailed Accuracy**: Provide a comprehensive answer. If the documentation lists steps, tools, or specific tolerances (e.g. torque, pressure), include ALL of them.
        - **Mandatory Inline Citations**: Every technical claim or step MUST be followed by a citation from the source provided in the context (e.g., [Manual X, Page 12]).
        - **Formatting**: Use **bolding** for technical values and "WARNING" blocks for safety-critical info.
        - **Strict Grounding**: Only use the provided context. If a value is missing, say it is not in the documentation.

        ### QUESTION
        {query}

        ### DETAILED RESPONSE
        """

        return {
            "prompt": prompt,
            "texts": texts,
        }

    def _generate_follow_up_questions(self, query: str) -> List[str]:
        """Generate concise follow-up questions for the response."""
        followup_prompt = (
            f"Based on the previous answer to \"{query}\", suggest 3 relevant follow-up questions "
            "that a user might ask.\nFormat your response as a numbered list (1. 2. 3.) with "
            "just the questions, no explanations."
        )

        followup_response = self.llm.invoke([HumanMessage(content=followup_prompt)])
        followup_text = (
            followup_response.content
            if hasattr(followup_response, "content")
            else str(followup_response)
        )

        follow_up_questions = []
        for line in followup_text.split("\n"):
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                question = line.split(".", 1)[1].strip() if "." in line else line
                if question:
                    follow_up_questions.append(question)

        return follow_up_questions[:3]
    
    def get_response(self, query: str, top_k: int = 3, equipment_name: str = None) -> dict:
        """
        Get a response to a user query using RAG.
        
        Args:
            query: User's question
            top_k: Number of relevant chunks to retrieve
            equipment_name: Filter results to specific equipment (optional)
        
        Returns dict with: answer, sources, follow_up_questions
        """
        payload = self._prepare_prompt_payload(query, top_k=top_k, equipment_name=equipment_name)
        if "error" in payload:
            return payload["error"]

        prompt = payload["prompt"]
        texts = payload["texts"]

        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, 'content') else str(response)
        follow_up_questions = self._generate_follow_up_questions(query)
        
        return {
            "answer": answer,
            "sources": [{"text": t[:200] + "..." if len(t) > 200 else t} for t in texts],
            "follow_up_questions": follow_up_questions[:3]
        }

    def stream_response(self, query: str, top_k: int = 3, equipment_name: str = None) -> Generator[Dict, None, None]:
        """Stream response events: token chunks, then final payload with sources/follow-ups."""
        payload = self._prepare_prompt_payload(query, top_k=top_k, equipment_name=equipment_name)
        if "error" in payload:
            yield {
                "type": "final",
                "answer": payload["error"]["answer"],
                "sources": payload["error"]["sources"],
                "follow_up_questions": payload["error"]["follow_up_questions"],
            }
            return

        prompt = payload["prompt"]
        texts = payload["texts"]
        answer_parts = []

        for chunk in self.llm.stream([HumanMessage(content=prompt)]):
            chunk_text = chunk.content if hasattr(chunk, "content") else str(chunk)
            if not chunk_text:
                continue
            answer_parts.append(chunk_text)
            yield {
                "type": "token",
                "content": chunk_text,
            }

        answer = "".join(answer_parts).strip()
        follow_up_questions = self._generate_follow_up_questions(query)
        yield {
            "type": "final",
            "answer": answer,
            "sources": [{"text": t[:200] + "..." if len(t) > 200 else t} for t in texts],
            "follow_up_questions": follow_up_questions,
        }
    
    def get_available_equipment(self) -> List[str]:
        """Get list of all equipment in the knowledge base"""
        if self.vectorstore is None:
            return []
        return self.vectorstore.get_available_equipment()
    
    def detect_equipment_from_query(self, query: str) -> str:
        """Detect equipment mentioned in the query text
        
        Returns:
            Equipment name if detected, None otherwise
        """
        if self.vectorstore is None:
            return None
        
        available_equipment = self.get_available_equipment()
        if not available_equipment:
            return None
        
        query_lower = query.lower()
        
        # Check for exact matches first (case-insensitive)
        for equipment in available_equipment:
            if equipment.lower() in query_lower:
                print(f"[INFO] Auto-detected equipment from query: {equipment}")
                return equipment
        
        # Check for partial matches (e.g., "Radar" matches "Radar_3D_BEL")
        # Split equipment names and check each part
        for equipment in available_equipment:
            parts = equipment.split('_')
            for part in parts:
                if len(part) > 3 and part.lower() in query_lower:
                    print(f"[INFO] Auto-detected equipment from query keyword '{part}': {equipment}")
                    return equipment
        
        return None
