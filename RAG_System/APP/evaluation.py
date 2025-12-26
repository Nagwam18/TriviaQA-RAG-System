import pandas as pd
import time
import re
from pipeline import run_rag

evaluation_set = [
    {"question": "Which number Beethoven symphony is known as 'The Pastoral'?", "ground_truth": "Sixth"},
    {"question": "Miami Beach in Florida borders which ocean?", "ground_truth": "Atlantic"},
    {"question": "What is the name of the perfume launched by British boyband JLS in January 2013?", "ground_truth": "Love"},
    {"question": "Caroline of Brunswick was the queen consort of which British King?", "ground_truth": "George IV"},
    {"question": "What is the official march of the Royal Navy?", "ground_truth": "Heart of Oak"},
    {"question": "Technically a shoal of fish becomes a school of fish when it is?", "ground_truth": "Swimming in the same direction"},
    {"question": "On which island was the famous photograph taken showing US Marines raising the US flag over Mt Suribachi in February 1945?", "ground_truth": "Iwo Jima"},
    {"question": "What was the first name of the character played by John Travolta in Saturday Night Fever?", "ground_truth": "Tony (Manero)"},
    {"question": "Jonas Salk developed a vaccine against what?", "ground_truth": "Polio"},
    {"question": "Who is said to have cut the Gordian Knot?", "ground_truth": "Alexander the Great"},
    {"question": "The Italian cheese called dolcelatte translates into English as what?", "ground_truth": "Sweet milk"},
    {"question": "What is the title of the last Harry Potter novel, published in 2007?", "ground_truth": "Harry Potter and the Deathly Hallows"},
    {"question": "Who was the first professional cricketer to captain England?", "ground_truth": "Len Hutton"},
    {"question": "Which country left the Commonwealth in 1972 and rejoined in 1989?", "ground_truth": "Pakistan"},
    {"question": "Wisent is an alternative name for which animal?", "ground_truth": "(European) Bison"},
    {"question": "In 1984, in Bophal, India, there was a leak of 30 tons of methyl isocyanate, which resulted in the deaths of 25,000 people. What American chemical company owned the plant where the leak occurred?", "ground_truth": "UNION CARBIDE"},
    {"question": "In which country is the annual International Alphorn Festival held?", "ground_truth": "SWITZERLAND"},
    {"question": "David Balfour and Alan Breck are characters in books by which author?", "ground_truth": "ROBERT LOUIS STEVENSON"},
    {"question": "High Willhays is the highest point of what National Park?", "ground_truth": "DARTMOOR"},
    {"question": "In 1973 the Paris Peace Accords were held in an attempt to end which war?", "ground_truth": "Vietnam"}
]

def normalize(text):
    """Remove punctuation, convert to lowercase, and strip spaces"""
    return re.sub(r'[^\w\s]', '', text).lower().strip() if text else ''

def evaluate_system(test_set):
    results = []
    latencies = []

    print(f"Starting Evaluation on {len(test_set)} questions...\n")

    for item in test_set:
        q = item["question"]
        gt = item["ground_truth"]

        res = run_rag(q)
        generated_ans = res.get("answer", "").strip()
        context = res.get("retrieved_context", "")
        latency = res.get("latency_ms", 0)

        latencies.append(latency)

        # Normalize for comparison
        gen_norm = normalize(generated_ans)
        gt_norm = normalize(gt)
        context_norm = normalize(context)

        relevance = "Yes" if gt_norm in context_norm else "No"
        context_ok = "Yes" if gt_norm in context_norm else "No"

        # Evaluate correctness
        if generated_ans == "Not found in context":
            status = "Incorrect"
        elif not context_ok:
            status = "Incorrect"
        elif gen_norm == gt_norm:
            status = "Correct"
        elif gt_norm in gen_norm or gen_norm in gt_norm:
            status = "Partially Correct"
        else:
            status = "Incorrect"

        results.append({
            "Question": q,
            "Ground Truth": gt,
            "RAG Answer": generated_ans,
            "Context Correct?": context_ok,
            "Evaluation Status": status,
            "Latency (ms)": latency,
            "Relevance": relevance
        })

    df = pd.DataFrame(results)
    accuracy = df["Evaluation Status"].isin(["Correct", "Partially Correct"]).sum() / len(df) * 100
    avg_latency = round(sum(latencies) / len(latencies), 2)
    relevance_pct = (df["Relevance"] == "Yes").sum() / len(df) * 100

    print("\n📊 Evaluation Summary")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Latency: {avg_latency} ms")
    print(f"Relevance: {relevance_pct:.2f}%")

    return df

# Run evaluation
df_final_results = evaluate_system(evaluation_set)
df_final_results
