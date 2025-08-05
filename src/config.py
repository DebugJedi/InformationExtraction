import os
os.environ["TRANSFORMERS_NO_TF"] ="1"
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import faiss
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
from typing import List
import logging
from src.utils import extract_text


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalRag:

    def __init__(self,
                 embedding_model: str = "all-miniLM-L6-v2",
                 llm_name: str ='mistralai/Mistral-7B-Instruct-v0.1',
                 device=None,
                 env_path: str = ".env",
                 raw_text: str = None
                 ):
       

        self._huggingface_login()
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.text_chunks = []
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, 
            device_map='auto', 
            torch_dtype= torch.float16 )
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if raw_text:
            self.build_knowledge_base(raw_text)
        
    def load_text_file(self, file_path: str) -> str:
        if ".pdf" in file_path:
            logger.info("Loading file....")
            text = extract_text.from_pdf(file_path)
            logger.info("Extraction completed....")
            return text
        elif ".txt" in file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
            
        else: return logger.info("Invalid file extension: Only pdf/txt file accepted!")

    def _load_env(self, path: str):
        if load_dotenv(dotenv_path=path):
            logger.info(f".env loaded from: {path}...")
        else:
            logger.warning(f".env file not found at: {path}...")

    def _huggingface_login(self):
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise ValueError("API Key not found in environment variables...")
        login(api_key)
        logger.info("Huggin Face login successful")

    def split_text(self, text, max_length=500):
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks=[]

        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk+= sentence + " "
            else:
                chunks.append(current_chunk.strip())
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def build_knowledge_base(self, raw_text):
        self.text_chunks = self.split_text(raw_text)
        embeddings = self.embedding_model.encode(self.text_chunks)
        dim=embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))
        logger.info("Knowledge base built successfully!")


    def retrieve_relevant_chunks(self, question, top_k = 5):
        question_embedding = self.embedding_model.encode([question])
        distance, indices = self.index.search(np.array(question_embedding), top_k)
        return [self.text_chunks[i] for i in indices[0]]
    
    def generate_answer(self, question, chunks= None):
        if chunks is None:
            chunks = self.retrieve_relevant_chunks(question)
        # context_chunks = self.retrieve_relevant_chunks(question)
        context = "\n".join(chunks)

        prompt = f"""You are an AI assistance. Use the context to answer the question. 
                context: {context}
                Question: {question}
                Answer: 
"""
        inputs = self.tokenizer(prompt, return_tensors = 'pt')
        inputs = {k: v.to(self.llm.device) for k,v in inputs.items()}
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens = 200,
            temperature = 0.5,
            do_sample = True,
            top_p = 0.95
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.split("Answer:")[-1].strip()

    def query(self, question, top_k=5):
        chunks=self.retrieve_relevant_chunks(question, top_k)
        return self.generate_answer(question, chunks)