import json, math

def hit_rate(rel, ret, k):
    return 1.0 if set(rel) & set(ret[:k]) else 0.0

def precision_at_k(rel, ret, k):
    top = ret[:k]
    return sum(d in rel for d in top) / k

def recall_at_k(rel, ret, k):
    return len(set(rel) & set(ret[:k])) / len(rel)

def reciprocal_rank(rel, ret):
    for i, d in enumerate(ret, 1):
        if d in rel:
            return 1.0 / i
    return 0.0

def average_precision(rel, ret):
    hits, score = 0, 0.0
    for i, d in enumerate(ret, 1):
        if d in rel:
            hits += 1
            score += hits / i
    return score / len(rel)

def ndcg_at_k(rel, ret, k):
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ret[:k], 1) if d in rel)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0

def evaluate(data, k=3):
    rows = []
    for d in data:
        rel, ret = d["relevant_doc_ids"], d["retrieved_doc_ids"]
        rows.append({
            "question": d["question"],
            f"hit@{k}": hit_rate(rel, ret, k),
            f"precision@{k}": precision_at_k(rel, ret, k),
            f"recall@{k}": recall_at_k(rel, ret, k),
            "rr": reciprocal_rank(rel, ret),
            "ap": average_precision(rel, ret),
            f"ndcg@{k}": ndcg_at_k(rel, ret, k),
        })
    keys = [c for c in rows[0] if c != "question"]
    summary = {("MRR" if c == "rr" else "MAP" if c == "ap" else c): sum(r[c] for r in rows) / len(rows) for c in keys}
    return rows, summary

if __name__ == "__main__":
    data = json.load(open("evals/ragas_law_dataset.json"))
    rows, summary = evaluate(data, k=3)
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"{k}: {v:.3f}")