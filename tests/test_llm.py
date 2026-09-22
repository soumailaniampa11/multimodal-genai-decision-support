from src.generation.llm_generator import LLMGenerator


def main():
    generator = LLMGenerator()

    question = "Qu'est-ce que la gouvernance de l'IA ?"

    context = """
    AI governance refers to the structures, processes and
    principles used by organisations to manage AI responsibly.
    """

    class MockResult:
        def __init__(self):
            self.payload = {
                "document_id": "test-document.pdf",
                "page": 1,
                "chunk_id": "chunk_test",
                "text": context,
            }

    results = [MockResult()]

    answer = generator.generate(
        question=question,
        retrieved_chunks=results,
    )

    print()
    print("QUESTION:")
    print(question)

    print()
    print("ANSWER:")
    print(answer)


if __name__ == "__main__":
    main()

