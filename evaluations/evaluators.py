import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
# Separate local evaluation model

judge_llm = ChatGoogleGenerativeAI(
    model="gemini-4-hydrogen",
    api_key=api_key,
    temperature=0
)

# Faithfulness evaluation
faithfulness_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator for a Retrieval Augmented Generation system.

Your task is to evaluate whether the generated answer is supported
by the retrieved context.

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluation rules:

1. Check every factual claim in the generated answer.
2. Determine whether the claims are supported by the context.
3. Penalize unsupported claims and hallucinations.
4. Do not penalize information that is correctly paraphrased.
5. If the context is empty, give a score of 0.

Return ONLY valid JSON in this format:

{{
    "score": 0.0,
    "reason": "Explain your evaluation briefly."
}}

Score must be between 0 and 1.
""")

# Answer correctness evaluation
answer_correctness_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator for a Retrieval Augmented Generation system.

Your task is to evaluate the correctness of a generated answer
by comparing it with the reference answer.

Question:
{question}

Reference Answer:
{reference_answer}

Generated Answer:
{generated_answer}

Evaluation rules:

1. Check whether the generated answer conveys the same
   essential information as the reference answer.
2. Accept correct paraphrasing and different wording.
3. Do not require an exact word-for-word match.
4. Penalize incorrect or contradictory information.
5. Penalize missing important information.
6. Do not penalize minor additional explanations if they
   are consistent with the reference answer.
7. Evaluate semantic correctness, not writing style.
8. Do not use external knowledge to invent facts.

Scoring guidelines:

1.0 = Fully correct and semantically equivalent.
0.8 = Mostly correct with minor missing details.
0.5 = Partially correct with important omissions.
0.2 = Mostly incorrect.
0.0 = Completely incorrect or irrelevant.

Return ONLY valid JSON:

{{
    "score": 0.0,
    "reason": "Explain your evaluation briefly."
}}

Score must be between 0 and 1.
""")

# Context precision evaluation : Context Precision measures how many of the retrieved documents are relevant to the user's question.
                            #    In simple words: Out of all the documents retrieved, how many were actually useful?   

context_precision_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator of RAG retrieval systems.

Your task is to evaluate the relevance of each retrieved
context to the user's question.

Question:
{question}

Retrieved Contexts:
{contexts}

Instructions:

1. Evaluate every retrieved context independently.
2. A context is relevant if it contains information useful
   for answering the user's question.
3. A context can be relevant even if it does not contain
   the complete answer.
4. Mark irrelevant or unrelated contexts as false.
5. Do not use external knowledge.
6. Do not judge the generated answer.
7. Do not reward a context merely because it contains
   similar keywords.

Calculate:
Context Precision = Number of relevant contexts /
                    Total number of retrieved contexts.

Return ONLY valid JSON:

{{
    "score": 0.0,
    "reason": "Explain your evaluation briefly.",
    "contexts": [
        {{
            "index": 1,
            "relevant": true,
            "reason": "Explanation"
        }}
    ]
}}

Score must be between 0 and 1.
""")


# Context Recall prompt : Context Recall measures whether the retrieved documents contain the information needed to answer the question.
                    #     In simple words: Did your retriever find all the information necessary to answer the question?

context_recall_prompt = ChatPromptTemplate.from_template("""
You are an expert evaluator of RAG retrieval systems.

Your task is to determine whether the retrieved contexts
contain the information required to answer the question.

Question:
{question}

Reference Answer:
{reference_answer}

Retrieved Contexts:
{contexts}

Instructions:

1. Break the reference answer into individual factual claims.
2. Check whether each claim is supported by at least
   one retrieved context.
3. Mark a claim as supported only when the retrieved
   contexts provide sufficient evidence.
4. Accept correct paraphrasing.
5. Do not use external knowledge.
6. Do not judge the generated answer.
7. Do not assume that a claim is retrieved simply
   because it appears in the reference answer.

Calculate:
Context Recall = Number of supported reference claims /
                 Total number of reference claims.

Return ONLY valid JSON:

{{
    "score": 0.0,
    "reason": "Explain your evaluation briefly.",
    "claims": [
        {{
            "claim": "Reference fact",
            "supported": true,
            "evidence": "Supporting retrieved context"
        }}
    ]
}}

Score must be between 0 and 1.
""")

def _response_text(content: object) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if isinstance(block, str):
                text_blocks.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_blocks.append(block["text"])
        if text_blocks:
            return "\n".join(text_blocks)

    raise ValueError(
        f"Judge returned unsupported content type: {type(content).__name__}"
    )

# faithfulness_evaluator function
def faithfulness_evaluator(inputs, outputs, reference_outputs=None):

    contexts = outputs.get("contexts", [])
    answer = outputs.get("answer", "")

    context = "\n\n".join(contexts)

    if not context.strip() or not answer.strip():
        return {
            "key": "faithfulness",
            "score": 0.0,
            "comment": "Missing context or generated answer."
        }

    messages = faithfulness_prompt.format_messages(
        context=context,
        answer=answer
    )

    response = judge_llm.invoke(messages)
    response_text = _response_text(response.content)

    try:
        # Extract JSON in case the model adds surrounding text
        match = re.search(
            r"\{.*\}",
            response_text,
            re.DOTALL
        )

        if not match:
            raise ValueError("Judge did not return valid JSON")

        result = json.loads(match.group())

        score = float(result["score"])

        if not 0 <= score <= 1:
            raise ValueError("Score must be between 0 and 1")

        return {
            "key": "faithfulness",
            "score": score,
            "comment": result["reason"]
        }

    except Exception as e:
        print("JUDGE EVALUATION ERROR:", e)
        print("RAW JUDGE RESPONSE:", response_text)
        raise
    
# answer_correctness_evaluator function
def answer_correctness_evaluator(
    inputs,
    outputs,
    reference_outputs=None
):

    question = inputs.get("question", "")
    generated_answer = outputs.get("answer", "")

    # Extract expected answer from LangSmith dataset
    reference_answer = (
        reference_outputs.get("answer", "")
        if reference_outputs else ""
    )

    if not generated_answer.strip() or not reference_answer.strip():
        return {
            "key": "answer_correctness",
            "score": 0.0,
            "comment": "Missing generated or reference answer."
        }

    messages = answer_correctness_prompt.format_messages(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer
    )

    response = judge_llm.invoke(messages)

    response_text = _response_text(response.content)

    try:
        match = re.search(
            r"\{.*\}",
            response_text,
            re.DOTALL
        )

        if not match:
            raise ValueError("Judge did not return valid JSON")

        result = json.loads(match.group())

        score = float(result["score"])

        if not 0 <= score <= 1:
            raise ValueError("Score must be between 0 and 1")

        return {
            "key": "answer_correctness",
            "score": score,
            "comment": result["reason"]
        }

    except Exception as e:
        print("ANSWER CORRECTNESS ERROR:", e)
        print("RAW JUDGE RESPONSE:", response_text)
        raise



def _run_context_evaluation(prompt, **kwargs):

    messages = prompt.format_messages(**kwargs)

    response = judge_llm.invoke(messages)

    response_text = _response_text(response.content)

    match = re.search(
        r"\{.*\}",
        response_text,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            f"Invalid judge JSON: {response_text}"
        )

    result = json.loads(match.group())

    score = float(result["score"])

    if not 0 <= score <= 1:
        raise ValueError("Score must be between 0 and 1")

    return result


def context_precision_evaluator(
    inputs,
    outputs,
    reference_outputs=None
):

    question = inputs.get("question", "")
    contexts = outputs.get("contexts", [])

    if not contexts:
        return {
            "key": "context_precision",
            "score": 0.0,
            "comment": "No retrieved contexts available."
        }

    result = _run_context_evaluation(
        context_precision_prompt,
        question=question,
        contexts="\n\n".join(
            f"Context {i+1}:\n{context}"
            for i, context in enumerate(contexts)
        )
    )

    return {
        "key": "context_precision",
        "score": result["score"],
        "comment": result.get("reason", "")
    }


def context_recall_evaluator(
    inputs,
    outputs,
    reference_outputs=None
):

    question = inputs.get("question", "")
    contexts = outputs.get("contexts", [])

    reference_answer = (
        reference_outputs.get("answer", "")
        if reference_outputs else ""
    )

    if not contexts or not reference_answer.strip():
        return {
            "key": "context_recall",
            "score": 0.0,
            "comment": "Missing contexts or reference answer."
        }

    result = _run_context_evaluation(
        context_recall_prompt,
        question=question,
        reference_answer=reference_answer,
        contexts="\n\n".join(
            f"Context {i+1}:\n{context}"
            for i, context in enumerate(contexts)
        )
    )

    return {
        "key": "context_recall",
        "score": result["score"],
        "comment": result.get("reason", "")
    }
