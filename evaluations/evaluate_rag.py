from langsmith import evaluate
from dotenv import load_dotenv
import os

from core.rag_chain import process_query
from evaluations.evaluators import (
    faithfulness_evaluator,
    answer_correctness_evaluator,
    context_precision_evaluator,
    context_recall_evaluator
)

load_dotenv()

EVAL_USER_ID = os.getenv("EVAL_USER_ID")

if not EVAL_USER_ID:
    raise ValueError("EVAL_USER_ID is missing from .env")


def rag_target(inputs: dict) -> dict:

    question = inputs["question"]

    result = process_query(
        question=question,
        user_id=EVAL_USER_ID
    )
    print("\n========== EVALUATION DEBUG ==========")

    print("Question:", question)

    print("Retrieved contexts:", len(result.get("contexts", [])))

    print("Answer:", result["answer"][:300])

    print("======================================\n")

    return {
        "answer": result["answer"],
        "contexts": result.get("contexts", []),
        "routed_to_local": result["routed_to_local"]
    }


results = evaluate(
    rag_target,
    data="secure-rag-public-evaluation",

    evaluators=[
        faithfulness_evaluator,
        answer_correctness_evaluator,
        context_precision_evaluator,
        context_recall_evaluator
    ],

    experiment_prefix="secure-rag-four-metric-evaluation"
)

print(results)