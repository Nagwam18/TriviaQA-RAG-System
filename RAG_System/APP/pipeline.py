import time
import re
from transformers import pipeline
from llm_model import model, tokenizer
from embedding_model import embedder
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate


# LLM pipeline 
raw_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,         
    temperature=0.0,            
    do_sample=False,
    return_full_text=False
)

llm_pipeline = HuggingFacePipeline(pipeline=raw_pipeline)


#vector DB
persist_dir = "./chroma_db"  
vector_db = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedder
)

retriever = vector_db.as_retriever(search_kwargs={"k":5})


# prompt 
prompt_template = """<s>[INST]
You are a STRICT answer-only bot.

Rules:
- Answer ONLY using the context
- Answer must be very short (1–3 words)
- If the answer is not explicitly present, reply EXACTLY with:
Not found in context
- No explanations
- No full sentences

Context:
{context}

Question:
{question}
[/INST]
Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)


# QA CHAIN  
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_pipeline,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
)


#  RAG Function
def run_rag(question: str):
    start_time = time.time()
    if question:
        question = question.strip()  
        question = re.sub(r'\s+', ' ', question)  

    # Handle empty or None queries 
    if not question or not question.strip():
        latency_ms = int((time.time() - start_time) * 1000)
        print(f"[WARNING] Empty or None query received. Latency: {latency_ms}ms")
        return {
            "question": question,
            "answer": "Empty query received",
            "retrieved_context": "",
            "latency_ms": latency_ms
        }

    try:
        # Retrieve context
        docs = retriever.invoke(question)
        context_string = "\n".join(doc.page_content.strip() for doc in docs)

        # Generate answer
        response = qa_chain.invoke({"query": question})
        raw_answer = response["result"].strip()
        answer = raw_answer.split("\n")[0].strip()

        # Clean answer 
        if (
            not answer
            or len(answer.split()) > 3
            or "not found" in answer.lower()
            or "context" in answer.lower()
            or "does not" in answer.lower()
        ):
            final_answer = "Not found in context"
        else:
            final_answer = answer.rstrip(".")

        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "question": question,
            "answer": final_answer,
            "retrieved_context": context_string,
            "latency_ms": latency_ms
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        print(f"[ERROR] Exception during processing: {str(e)}. Latency: {latency_ms}ms")
        return {
            "question": question,
            "answer": "Not found in context",
            "retrieved_context": "",
            "latency_ms": latency_ms
        }


# TEST 
if __name__ == "__main__":
    query = "	Which country left the Commonwealthin 1972 and rejoined in 1989?"
    result = run_rag(query)
    
    print("--- RAG Result ---")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Latency: {result['latency_ms']} ms")
    print("\n--- Context Found (Snippets) ---")
    print(result["retrieved_context"][:500])
