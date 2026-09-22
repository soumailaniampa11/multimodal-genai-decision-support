import json

from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.qdrant_store import QdrantStore
from src.generation.llm_generator import LLMGenerator
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.evaluation.llm_evaluator import LLMEvaluator


EVALUATION_FILE = "data/evaluation/rag_questions.json"
TOP_K = 5


def load_evaluation_questions():

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("No evaluation questions found.")

    return questions


def main():

    questions = load_evaluation_questions()

    print()
    print("=" * 80)
    print("RAG SYSTEM - FULL EVALUATION")
    print("=" * 80)

    print()
    print(f"Evaluation questions: {len(questions)}")
    print(f"Top-K retrieval: {TOP_K}")

    # ============================================================
    # INITIALIZE COMPONENTS ONCE
    # ============================================================

    embedding_model = EmbeddingModel()
    store = QdrantStore()
    generator = LLMGenerator()
    retrieval_evaluator = RetrievalEvaluator()
    llm_evaluator = LLMEvaluator()

    # ============================================================
    # GLOBAL RESULTS
    # ============================================================

    all_results = []

    # ============================================================
    # EVALUATE EACH QUESTION
    # ============================================================

    for question_number, evaluation in enumerate(
        questions,
        start=1,
    ):

        question_id = evaluation["id"]
        question = evaluation["question"]
        relevant_pages = evaluation["relevant_pages"]
        reference = evaluation["reference_answer"]

        print()
        print()
        print("#" * 80)
        print(
            f"QUESTION {question_number}/{len(questions)} "
            f"- {question_id}"
        )
        print("#" * 80)

        print()
        print("QUESTION:")
        print(question)

        print()
        print("GROUND-TRUTH PAGES:")
        print(relevant_pages)

        # ========================================================
        # RETRIEVAL
        # ========================================================

        print()
        print("-" * 80)
        print("RETRIEVAL")
        print("-" * 80)

        query_vector = embedding_model.model.encode(
            question
        ).tolist()

        results = store.search(
            query_vector=query_vector,
            limit=TOP_K,
        )

        print()
        print(f"Retrieved {len(results)} chunks:")

        for rank, result in enumerate(
            results,
            start=1,
        ):

            payload = result.payload

            print(
                f"#{rank} | "
                f"Page: {payload['page']} | "
                f"Chunk: {payload['chunk_id']} | "
                f"Score: {result.score:.4f}"
            )

        # ========================================================
        # RETRIEVAL EVALUATION
        # ========================================================

        precision_at_k = retrieval_evaluator.precision_at_k(
            results=results,
            relevant_pages=relevant_pages,
            k=TOP_K,
        )

        print()
        print("RETRIEVAL METRICS")
        print(
            f"Precision@{TOP_K}: "
            f"{precision_at_k:.3f}"
        )

        # ========================================================
        # GENERATION
        # ========================================================

        print()
        print("-" * 80)
        print("GENERATION")
        print("-" * 80)

        answer = generator.generate(
            question=question,
            retrieved_chunks=results,
        )

        print()
        print("GENERATED ANSWER:")
        print(answer)

        # ========================================================
        # LLM EVALUATION
        # ========================================================

        print()
        print("-" * 80)
        print("LLM EVALUATION")
        print("-" * 80)

        contexts = [
            result.payload["text"]
            for result in results
        ]

        llm_scores = llm_evaluator.evaluate(
            question=question,
            answer=answer,
            contexts=contexts,
            reference=reference,
        )

        for metric, data in llm_scores.items():

            print()
            print(metric.upper())
            print(
                f"Score : {data['score']:.3f}"
            )
            print(
                f"Reason: {data['reason']}"
            )

        # ========================================================
        # STORE RESULTS
        # ========================================================

        all_results.append(
            {
                "id": question_id,
                "precision_at_k": precision_at_k,
                "faithfulness": llm_scores[
                    "faithfulness"
                ]["score"],
                "context_relevance": llm_scores[
                    "context_relevance"
                ]["score"],
                "answer_relevance": llm_scores[
                    "answer_relevance"
                ]["score"],
                "reference_alignment": llm_scores[
                    "reference_alignment"
                ]["score"],
            }
        )

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================

    number_of_questions = len(all_results)

    average_precision = sum(
        result["precision_at_k"]
        for result in all_results
    ) / number_of_questions

    average_faithfulness = sum(
        result["faithfulness"]
        for result in all_results
    ) / number_of_questions

    average_context_relevance = sum(
        result["context_relevance"]
        for result in all_results
    ) / number_of_questions

    average_answer_relevance = sum(
        result["answer_relevance"]
        for result in all_results
    ) / number_of_questions

    average_reference_alignment = sum(
        result["reference_alignment"]
        for result in all_results
    ) / number_of_questions

    # ============================================================
    # FINAL RESULTS
    # ============================================================

    print()
    print()
    print("=" * 80)
    print("FINAL EVALUATION RESULTS")
    print("=" * 80)

    print()
    print(
        f"Questions evaluated       : "
        f"{number_of_questions}"
    )

    print(
        f"Average Precision@{TOP_K}     : "
        f"{average_precision:.3f}"
    )

    print(
        f"Average Faithfulness     : "
        f"{average_faithfulness:.3f}"
    )

    print(
        f"Average Context Relevance: "
        f"{average_context_relevance:.3f}"
    )

    print(
        f"Average Answer Relevance : "
        f"{average_answer_relevance:.3f}"
    )

    print(
        f"Average Reference Alignment: "
        f"{average_reference_alignment:.3f}"
    )

    # ============================================================
    # QUESTION-BY-QUESTION SUMMARY
    # ============================================================

    print()
    print("=" * 80)
    print("QUESTION-BY-QUESTION SUMMARY")
    print("=" * 80)

    print()

    print(
        f"{'ID':<8}"
        f"{'P@5':<10}"
        f"{'Faith.':<10}"
        f"{'Ctx Rel.':<10}"
        f"{'Ans Rel.':<10}"
        f"{'Ref Align.':<12}"
    )

    print("-" * 80)

    for result in all_results:

        print(
            f"{result['id']:<8}"
            f"{result['precision_at_k']:<10.3f}"
            f"{result['faithfulness']:<10.3f}"
            f"{result['context_relevance']:<10.3f}"
            f"{result['answer_relevance']:<10.3f}"
            f"{result['reference_alignment']:<12.3f}"
        )

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()