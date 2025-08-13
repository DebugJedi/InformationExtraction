import os
import re 
import logging
from typing import List, Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import faiss
import torch
import requests
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_tranformers import SentenceTransformer
from src.utils import extract_text


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


HF_INFERENCE_URL = "https://api-inference.huggingface.co/models"
DEFAULT_EMBED_MODEL = "all-miniLM-L6-v2"
DEFAULT_GEN_MODEL = "microsoft/phi-2"
DEFAULT_MAX_CONTEXT_CHARS = 2000
DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_TIMEOUT_S = 20

_HTTP = requests.Session()

class LocalRag:
    """
    Lightweight RAG with:
        - sentence-transformers embeddings (batched, normalized)
        - FAISS retrieval (inner product ~ cosine)
        - HF Inference API for generation (no local LLM spin-up)
        - Optional index persistence (faiss.write/read)
    """

    def __init__(
        self,
        embedding_models: str = DEFAULT_EMBED_MODEL,
        gen_model: str = DEFAULT_GEN_MODEL,
        device: Optional[str] = None,
        env_path: str = ".env",
        raw_text: Optional[str] = None,
        index_path: Optional[str] = None,
        chunks_path: Optional[str] = None,
    ):

        self._load_env(env_path)
        self._huggingface_login()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTranformer(embedding_model, device=self.device)

        self.index: Optional[faiss.Index] = None
        self.text_chunks: List[str] = []


        self.gen_model = gen_model
        self.hf_token = os.getenv("HF_API_KEY")
        if not self.hf_token:
            raise ValueError("HF_API_KEY missing. Add it to your environment/.env")

        if index_path and chunks_path and self._load_index(index_path, chunks_path):
            logger.info("Loaded persisted FAISS index and chunks.")

        elif raw_text:
            self.build_knowledge_base(raw_text)


    #  IO / ENV
    def _load_env(self, path: str) -> None:
        if load_dotenv(dotenv_path = path):
            logger.info(f".env loaded from: {path}")

        else: 
            logger.warning(f".env file not found at: {path} ")

    def _huggingface_login(self) -> None:
        """
        Optional: logs in so future gated downloads (if any) just work. 
        Not strickly required for HF Interece API, but harmless + usefull.
        """
        token = os.getenv("HF_API_KEY")
        if token:
            try:
                login(token)
                logger.info("Hugging Face login successful.")
            except Exception as e:
                logger.warning(f"HF login failed (continuing anyway): {e}")
        else:
            logger.warning("HF_API_KEY not found for login (continuing).")

    def load_text_file(self, file_path: str) -> str:
        """
        Load text from .pdf or .txt. Returns extracted text.
        """

        fp = file_path.lower()

        if fp.endswith(".pdf"):
            logger.info("Extracting text from PDF...")
            text = extract_text.from_pdf(file_path)
            logger.info("PDF extraction completed...")
            return text

        elif fp.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError("Invalid file extension. Only .pdf or .txt supported. ")

    @staticmethod
    def split_text(text: str, max_length: int = 500) -> List[str]:
        """
        Sentence-aware chunking with a hard char cap per chunk.
        Avoids dropping sentences and minimizes string concatenations. 
        """ 

        # Normalize whitespace once
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, cur, cur_len = [], [], 0

        for s in sentences:
            sl = len(s)
            if cur_len + sl +1 <= max_length:
                cur.append(s)
                cur_len +=sl+1
            else: 
                if cur:
                    chunks.append(" ".join(cur))
                cur = [s]
                cur_len = sl

        if cur: 
            chunks.append(" ".join(cur))
        
        return chunks

    # Build / Persist Index 
    def build_knowledge_base(self, raw_text: str) -> None:
        """
        Build FAISS index using normalized embeddings (cosine via inner product).
        """

        self.text_chunks = self.split_text(raw_text)

        if not self.text_chunks:
            raise ValueError("No text chuks produced from input.")
        
        embeds = self.embedding_model.encode(
            self.text_chunks,
            batch_size=DEFAULT_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        dim = embeds.shape[1]
        index = faiss.IndexFlatIP(dim) # IP on normalized vectors == cosine
        index.add(embeds)

        self.index = index
        logger.info(f"Knowledge base built. Chunks: {len(self.text_chunks)}")

    
    def save_index(self, index_path: str, chunks_path: str) -> None:
        """
        Persist index and chunk list to disk.
        """
        if self.index is None or not self.text_chunks:
            raise RuntimeError("No index/chunks to save.")
        faiss.write_index(self.index, index_path)
        np.save(chunks_path, np.array(self.text_chunks, dtype=object))
        logger.info(f"Saved index to {index_path} and chunks to {chunks_path}.")

    def _load_index(self, index_path: str, chunks_path: str) -> bool:
        """
        Try to load persisted index. Returns True on success.
        """

        try:
            if not (os.path.exists(index_path)) and os.path.exists(chunks_path):
                return False
            self.index = faiss.read_index(index_path)
            self.text_chunks = np.load(chunks_path, allow_pickle=True).tolist()
            return True
        except Exception as e:
            logger.warning(f"Failed to load persisted index: {e}")
            self.index, self.text_chunks = None, []
            return False

    def retrieve_relevant_chunks(self, question: str, top_k: int=DEFAULT_TOP_K) -> List[str]:
        if self.index is None:
            raise RuntimeError("Index not built. Provide raw_text or load a saved index.")
        q = self.embedding_model.encode(
            [question],
            convert_to_numpy = True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        _, idx = self.index.search(q, top_k)
        return [self.text_chunks[i] for i in idx[0]]
    

    def _generate_with_hf(
            self,
            prompt: str,
            max_new_tokens: int = 200,
            temperature: float = 0.5,
            stop: Optional[List[str]] = None,
    ) -> str:
        """
        Hit HF Interface API with pooled session and reasonable timeout.
        """

        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature
            },
            "options": {"wait_for_model": True}, #Cold start safety.
        }
        if stop:
            payload["parameters"]["stop"] = stop

        url = f"{HF_INFERENCE_URL}/{self.gen_model}"

        resp = _HTTP.post(url, headers=headers, json= payload, timeout=DEFAULT_TIMEOUT_S )
        resp.raise_for_status() #will raise HTTPError on non-2xx

        data = resp.json()

        # HF can return either a list of dicts or a dict with 'generated_text'
        if isinstance(data, list) and data and "generated_text" in data:
            return data["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        # Some models return {'error': '...'} or different shapes
        raise RuntimeError(f"Unexpected HF response: {data}")
    

    def generate_answer(
            self, 
            question: str,
            chunks: Optional[List[str]] = None,
            max_context_chars: int= DEFAULT_MAX_CONTEXT_CHARS,
            max_new_tokens: int = 200,
            temperature: float = 0.5,
    ) -> str:
        if chunks is None:
            chunks = self.retrieve_relevant_chunks(question)
        
        # Build compact context: hard cap length to keep generation snappy
        context = '\n'.join(chunks)[:max_context_chars]

        prompt = (
            "You are a concise AI assistant. Use the provided context to answer the qeustion. \n"
            "IF the answer isn't in the context, say you don't know. \n\n"
            f"Context: \n{context}\n\n"
            f"Question: {question}\n"
            "Answer: "
        )

        try:
            text = self._generate_with_hf(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop=["\n\nQuestion: "], #mild guard
            )
        except requests.HTTPError as e:
            logger.error(f"HF Inference API HTTP error: {e} / {getattr(e, 'response', None)}")
            raise
        except Exception as e:
            logger.error(f"HF Inference API error: {e}")
            raise

        answer = text.split("Answer: ")[-1].strip()

        return answer
    

    def query(self, question: str, top_k: int = DEFAULT_TOP_K) -> str:
        chunks = self.retrieve_relevant_chunks(question, top_k=top_k)
        return self.generate_answer(question=question, chunks=chunks)