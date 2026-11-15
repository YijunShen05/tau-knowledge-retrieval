"""Run with python replay_bm25.py /path/to/tau2-bench (rank-bm25 required)."""
import json
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi

def main(repo):
    domain = repo / "data/tau2/domains/banking_knowledge"
    docs = [json.loads(path.read_text()) for path in sorted((domain / "documents").glob("*.json"))]
    index = BM25Okapi([doc["content"].lower().split() for doc in docs])
    queries = ["Referral bonus offerings, highest bonus", "credit card referral bonus", "checking account referral bonus"]
    # Task labels are consulted only after all rankings have been computed.
    rankings = {}
    for query in queries:
        scores = index.get_scores(query.lower().split())
        indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:10]
        rankings[query] = [docs[i] for i in indices]
    task = json.loads((domain / "tasks/task_098.json").read_text())
    gold = set(task["required_documents"])
    print(f"Corpus: {len(docs)} documents. Task 098: {len(gold)} gold documents.")
    for query, hits in rankings.items():
        matched = sum(doc["id"] in gold or doc["title"] in gold for doc in hits)
        print(f"{query}: {matched}/{len(gold)} task documents in top 10")
        for rank, doc in enumerate(hits, 1):
            print(f"{rank:2d}  {doc['id']}  {doc['title']}")

if __name__ == "__main__":
    main(Path(sys.argv[1]))
