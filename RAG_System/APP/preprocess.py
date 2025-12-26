import os
import json
from datasets import load_dataset
try:
    from embedding_model import embedder
    from llm_model import tokenizer
    print("[SUCCESS] Imported embedder and tokenizer")
except ImportError:
    print("[ERROR] Could not find embedding_model.py or llm_model.py. Make sure to run those cells first!")

print("Tokenizer loaded successfully")


def load_triviaqa_subset(n_samples=700):
    print(f"[load_triviaqa_subset] Loading {n_samples} samples from TriviaQA...")
    dataset = load_dataset("mandarjoshi/trivia_qa", "unfiltered", split=f"train[:{n_samples}]")
    docs = []

    for item in dataset:
        sr = item.get("search_results", {})
        contexts = sr.get("search_context", [])

        for ctx in contexts:
            if ctx and ctx.strip():
                docs.append({
                    "doc_id": item["question_id"],
                    "text": ctx.strip()
                })

    print(f"[load_triviaqa_subset] Docs extracted: {len(docs)}")
    # if docs:
    #     print("[load_triviaqa_subset] Example doc:", docs[0])
    return docs

import re

def clean_text(text, min_length=15):
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    if len(cleaned) < min_length:
        return None

    return cleaned

def chunk_text(text, min_tokens=50, max_tokens=256, overlap=50): 
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) < max_tokens:
       return [tokenizer.decode(token_ids)]
    
    chunks = []
    start = 0
    while start < len(token_ids):
        end = start + max_tokens
        chunk_ids = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids))
        start += max_tokens - overlap 
    return chunks

def preprocess_triviaqa(n_samples=700, min_tokens_chunk=50, max_tokens_chunk=256, overlap_chunk=50):
    print(f"[preprocess_triviaqa] Starting preprocessing for {n_samples} samples...")
    docs = load_triviaqa_subset(n_samples)
    final_chunks = []

    for idx, d in enumerate(docs):
        cleaned = clean_text(d["text"])
        if not cleaned:
            print(f"[clean_text] Doc {d['doc_id']} skipped (too short)")
            continue

        chunks = chunk_text(cleaned, min_tokens=min_tokens_chunk, max_tokens=max_tokens_chunk, overlap=overlap_chunk)
        if not chunks:
            print(f"[chunk_text] Doc {d['doc_id']} produced 0 chunks")
            continue

        for i, ch in enumerate(chunks):
            final_chunks.append({
                "doc_id": d["doc_id"],
                "chunk_id": i,
                "text": ch
            })
        if idx < 3: 
            print(f"[preprocess_triviaqa] Doc {d['doc_id']} → {len(chunks)} chunks")

    print(f"[preprocess_triviaqa] Total chunks generated: {len(final_chunks)}")
    if final_chunks:
        print("[preprocess_triviaqa] Example chunk:", final_chunks[0])
    return final_chunks

def store_chunks_metadata(final_chunks, output_file="chunks_metadata.json"):
    metadata_list = []

    for chunk in final_chunks:
        text = chunk.get("text", "")
        meta = {
            "doc_id": chunk.get("doc_id"),
            "chunk_id": chunk.get("chunk_id"),
            "text": text,  
            "text_length": len(text),
            "num_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"])
        }
        metadata_list.append(meta)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    
    print(f"[store_chunks_metadata] Stored metadata for {len(metadata_list)} chunks in {output_file}")
    return metadata_list




#TEST
if __name__ == "__main__":
    print("[main] Running full preprocessing test...")
    final_chunks = preprocess_triviaqa(n_samples=700, min_tokens_chunk=50, max_tokens_chunk=256, overlap_chunk=50)

    print(f"[main] Test complete. Total chunks generated: {len(final_chunks)}")
    if final_chunks:
        print("[main] Example chunk:", final_chunks[0])
    chunks_metadata = store_chunks_metadata(final_chunks)
    print(chunks_metadata[:2])  