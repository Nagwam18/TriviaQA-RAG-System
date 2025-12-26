import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv() 
hf_token = os.getenv("HF_TOKEN")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cuda'} 
)

print(f"[INFO] Embedding model '{MODEL_NAME}' loaded")