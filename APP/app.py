import time
import json
import requests
import nest_asyncio
import uvicorn
from threading import Thread
from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import run_rag   

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_context: str 
    latency_ms: int

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    result = run_rag(req.question)
    return QueryResponse(
        question=result.get("question", req.question),
        answer=result.get("answer", "No answer generated"),
        retrieved_context=result.get("retrieved_context", ""),
        latency_ms=result.get("latency_ms", 0)
    )

nest_asyncio.apply()

def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="error")

thread = Thread(target=run_api, daemon=True)
thread.start()

print("[INFO] Waiting for server to stabilize...")
time.sleep(15) 

url = "http://127.0.0.1:8050/query"
payload = {"question": "Which country left the Commonwealthin 1972 and rejoined in 1989?"}

try:
    resp = requests.post(url, json=payload)
    print("POST status:", resp.status_code)
    if resp.status_code == 200:
        print("POST response:")
        print(json.dumps(resp.json(), indent=4))
    else:
        print("Error Response:", resp.text)
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
