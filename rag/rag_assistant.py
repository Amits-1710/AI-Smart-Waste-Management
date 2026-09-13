from .retriever import retrieve_documents
from .knowledge_base import get_documents_by_category


def build_context(results):
    """
    Build context from retrieved documents.
    """

    context_parts = []

    for result in results:

        document = result["document"]

        title = document.get(
            "title",
            "Unknown"
        )

        category = document.get(
            "category",
            "Unknown"
        )

        content = document.get(
            "content",
            ""
        )

        context_parts.append(
            f"Title: {title}\n"
            f"Category: {category}\n"
            f"Content: {content}"
        )

    return "\n\n".join(context_parts)


def generate_answer(query, results):
    """
    Generate an answer using retrieved knowledge.
    """

    if not results:

        return (
            "I could not find relevant "
            "waste-management information "
            "in the knowledge base."
        )

    top_document = results[0]["document"]

    content = top_document.get(
        "content",
        "No guidance available."
    )

    category = top_document.get(
        "category",
        "Unknown"
    )

    answer = (
        "Based on the available "
        "waste-management information:\n\n"
        f"{content}\n\n"
        f"Category: {category}\n\n"
        "Note: Waste-management rules can "
        "vary by location. Follow applicable "
        "local collection and disposal guidelines."
    )

    return answer


def ask_assistant(query, top_k=3):
    """
    Retrieve documents and generate an answer.
    """

    results = retrieve_documents(
        query,
        top_k=top_k
    )

    answer = generate_answer(
        query,
        results
    )

    return {
        "query": query,
        "results": results,
        "answer": answer
    }


def display_response(response):
    """
    Display RAG assistant response.
    """

    print("\n" + "=" * 60)
    print("RAG SUSTAINABILITY ASSISTANT")
    print("=" * 60)

    print("\nUser Question:")
    print(response["query"])

    print("\n" + "-" * 60)
    print("Retrieved Information")
    print("-" * 60)

    for i, result in enumerate(
        response["results"],
        start=1
    ):

        document = result["document"]

        print(
            f"\n{i}. "
            f"{document.get('title', 'Unknown')}"
            f" | Score: {result['score']}"
        )

    print("\n" + "-" * 60)
    print("ASSISTANT ANSWER")
    print("-" * 60)

    print("\n" + response["answer"])


def get_prediction_guidance(predicted_class):
    """
    Get waste-management guidance
    based on AI-predicted waste class.
    """

    category_mapping = {

        "battery": "battery",

        "biological": "organic",

        "brown-glass": "glass",

        "green-glass": "glass",

        "white-glass": "glass",

        "cardboard": "cardboard",

        "clothes": "textile",

        "shoes": "textile",

        "metal": "metal",

        "paper": "paper",

        "plastic": "plastic",

        "trash": "general"
    }

    predicted_class = (
        predicted_class
        .lower()
        .strip()
    )

    category = category_mapping.get(
        predicted_class,
        predicted_class
    )

    documents = get_documents_by_category(
        category
    )

    if not documents:

        return {
            "category": category,
            "title": "No specific guidance available",
            "content": (
                "No specific knowledge-base "
                "document was found for this prediction."
            )
        }

    document = documents[0]

    return {
        "category": category,
        "title": document.get(
            "title",
            "Waste Management Guidance"
        ),
        "content": document.get(
            "content",
            "No guidance available."
        )
    }


if __name__ == "__main__":

    print("=" * 60)
    print("AI SMART WASTE MANAGEMENT")
    print("PHASE 8.3 - RAG SUSTAINABILITY ASSISTANT")
    print("=" * 60)

    question = (
        "How should I dispose plastic waste?"
    )

    response = ask_assistant(
        question,
        top_k=3
    )

    display_response(response)

    print("\n" + "=" * 60)
    print("PHASE 8.5 - PREDICTION GUIDANCE TEST")
    print("=" * 60)

    predicted_class = "plastic"

    guidance = get_prediction_guidance(
        predicted_class
    )

    print(
        f"\nPredicted Class: "
        f"{predicted_class}"
    )

    print(
        f"Knowledge Category: "
        f"{guidance['category']}"
    )

    print(
        f"\nKnowledge Title: "
        f"{guidance['title']}"
    )

    print(
        f"\nGuidance:\n"
        f"{guidance['content']}"
    )

    print(
        "\nRAG assistant test completed."
    )