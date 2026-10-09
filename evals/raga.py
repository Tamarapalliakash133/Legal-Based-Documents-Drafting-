import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness,answer_relevancy,context_precision,context_recall
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

datas = json.load(open("evals/ragas_law_dataset.json"))

ds = Dataset.from_list([
    {
        "question" : data["question"],
        "answer" : data["answer"],
        "contexts" : data["contexts"],
        "ground_truth" : data["ground_truth"]
    }
    for data in datas
])

ds_small = ds.select(range(min(5, len(ds))))

langchain_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

print("Embedding test:", len(embeddings.embed_query("Test query")))

answer_relevancy.embeddings = embeddings

result = evaluate(ds_small,metrics = [faithfulness,answer_relevancy,context_recall,context_precision])

print(result)