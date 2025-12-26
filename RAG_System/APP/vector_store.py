import json
import uuid
import os
import shutil
from langchain_community.vectorstores import Chroma
from embedding_model import embedder 

persist_dir = "./chroma_db"  
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
    print(f"[INFO] Deleted old database at {persist_dir}")

if not os.path.exists("chunks_metadata.json"):
    print("[ERROR] chunks_metadata.json not found! Please run preprocess.py first.")
else:
    with open("chunks_metadata.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[INFO] Loaded {len(chunks)} chunks from JSON")

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedder
    )

    texts = [chunk["text"] for chunk in chunks]
    ids = [f'{chunk["doc_id"]}_{chunk["chunk_id"]}_{uuid.uuid4().hex}' for chunk in chunks]

    print(f"[INFO] Generating embeddings for {len(texts)} texts using GPU...")
    all_embeddings = embedder.embed_documents(texts)

    batch_size = 2000
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_ids = ids[start:end]
        batch_texts = texts[start:end]
        batch_embeddings = all_embeddings[start:end]

        vectorstore._collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings
        )
        print(f"[INFO] Stored batch {start}–{min(end, len(texts))} items")

    vectorstore.persist()
    print(f"[INFO] Stored total {len(texts)} embeddings in ChromaDB at '{persist_dir}'")