
import json
import os
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams

load_dotenv()

DATASET_PATH = "evals/evals_data/ragas_law_dataset.json"

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)


dataset = dataset[:5]

test_cases = []

for item in dataset:
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=item["answer"],
        expected_output=item["ground_truth"],
        retrieval_context=item["contexts"],
        context=item["contexts"],
    )
    test_cases.append(test_case)


faithfulness = FaithfulnessMetric(
    threshold=0.7,
    include_reason=True,
)

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    include_reason=True,
)


contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    include_reason=True,
)


answer_correctness = GEval(
    name="Answer Correctness",
    criteria=(
        "Evaluate whether the actual output is factually correct "
        "compared with the expected output. Consider semantic equivalence "
        "rather than requiring exact wording. Penalize incorrect or "
        "contradictory claims."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)


answer_completeness = GEval(
    name="Answer Completeness",
    criteria=(
        "Evaluate whether the actual output covers all important facts, "
        "conditions, and elements required by the expected output. "
        "Penalize missing material information, but do not require "
        "unnecessary details."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)


hallucination = HallucinationMetric(
    threshold=0.5,
    include_reason=True,
)

metrics = [
    faithfulness,
    answer_relevancy,
    contextual_relevancy,
    answer_correctness,
    answer_completeness,
    hallucination,
]


results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
)


print(results)
