import os
import pdfplumber
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import faiss
import joblib
import torch
from typing import List, Tuple

class ResumeRAG:
    def __init__(self):
        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-llm-7b-chat",
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            device_map="auto"
        )
        self.index = None
        self.chunks = []
        self.conversation_history = []

    def extract_text(self, pdf_path: str) -> str:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    def chunk_resume(self, text: str) -> List[str]:
        # Split by common resume section headers
        sections = re.split(
            r'(?i)(?:\n\s*)(Experience|Education|Skills|Projects|Summary|Objective|Achievements|Certifications|Work History|Technical Skills|Languages|Interests)\s*(?::|\n)',
            text
        )[1:]
        
        chunks = []
        for i in range(0, len(sections), 2):
            if i+1 < len(sections):
                header = sections[i].strip()
                content = sections[i+1].strip()
                chunks.append(f"{header}: {content}")
        
        # Fallback for non-standard resumes
        if not chunks:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            
        return chunks

    def process_resume(self, pdf_path: str):
        text = self.extract_text(pdf_path)
        self.chunks = self.chunk_resume(text)
        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=False)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Cache embeddings
        cache_path = pdf_path.replace('.pdf', '.cache')
        joblib.dump((self.chunks, embeddings), cache_path)

    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        return [self.chunks[i] for i in indices[0]]

    def generate_prompt(self, question: str, context: str) -> str:
        history = "\n".join([f"### Human: {q}\n### Assistant: {a}" for q, a in self.conversation_history[-2:]])
        return f"""
        <｜begin▁of▁sentence｜>You are an expert resume analyst. 
        Use only the following resume context to answer questions. 
        If the answer isn't in the context, say 'I don't see that in the resume.'
        
        Context:
        {context}
        
        {history}
        ### Human: {question}
        ### Assistant:
        """

    def ask(self, question: str) -> str:
        if not self.index:
            return "Please upload a resume first"
        
        context_chunks = self.retrieve_relevant_chunks(question)
        context = "\n\n".join(context_chunks)
        prompt = self.generate_prompt(question, context)
        
        response = self.llm_pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            eos_token_id=self.llm_tokenizer.eos_token_id,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )[0]['generated_text']
        
        # Extract only the assistant's response
        answer = response.split("### Assistant:")[-1].strip()
        
        # Update conversation history
        self.conversation_history.append((question, answer))
        return answer