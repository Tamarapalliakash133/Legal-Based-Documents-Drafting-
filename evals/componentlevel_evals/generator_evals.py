import json
import os
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric,ContextualRecallMetric

load_dotenv()


with open("evals/evals_data/ragas_law_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)


data_small = data[:5]


test_cases = []

for item in data_small:
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=item["answer"],
        retrieval_context=item["contexts"],
        expected_output=item["ground_truth"],
    )
    test_cases.append(test_case)


metrics = [
    FaithfulnessMetric(
        threshold=0.7,
        include_reason=True,
    ),
    AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True,
    ),
    ContextualPrecisionMetric(
        threshold=0.7,
        include_reason=True,
    ),
    ContextualRecallMetric(
        threshold=0.7,
        include_reason=True,
    ),
]


results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
)

print(results)


