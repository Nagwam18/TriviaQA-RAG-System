import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dotenv import load_dotenv
import os

load_dotenv() 
hf_token = os.getenv("HF_TOKEN")


LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"[INFO] Loading tokenizer for Mistral...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, token=hf_token)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[INFO] Loading Mistral-7B-v0.2 in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token
)

print(f"[INFO] LLM model loaded successfully on {model.device}")
