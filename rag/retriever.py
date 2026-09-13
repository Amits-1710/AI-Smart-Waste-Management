from .knowledge_base import get_all_documents


def tokenize(text):
    """
    Convert text into a set of useful lowercase words.
    """

    text = text.lower()

    punctuation = ",.!?;:\"'()[]{}-/"

    for char in punctuation:
        text = text.replace(char, " ")

    words = text.split()

    stop_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "am",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "how",
        "what",
        "where",
        "can",
        "should",
        "i",
        "my",
        "do"
    }

    return {
        word
        for word in words
        if word not in stop_words
    }


def calculate_score(query, document):
    """
    Calculate relevance score between query and document.
    """

    query_words = tokenize(query)

    title = document.get("title", "")
    content = document.get("content", "")

    keywords = document.get("keywords", [])

    keyword_text = " ".join(
        str(keyword) for keyword in keywords
    )

    title_words = tokenize(title)
    content_words = tokenize(content)
    keyword_words = tokenize(keyword_text)

    score = 0

    # Title match gets higher weight
    score += len(
        query_words.intersection(title_words)
    ) * 3

    # Keyword match
    score += len(
        query_words.intersection(keyword_words)
    ) * 2

    # Content match
    score += len(
        query_words.intersection(content_words)
    )

    return score


def retrieve_documents(query, top_k=3):
    """
    Retrieve the most relevant documents.
    """

    documents = get_all_documents()

    scored_documents = []

    for document in documents:

        score = calculate_score(
            query,
            document
        )

        if score > 0:

            scored_documents.append(
                {
                    "document": document,
                    "score": score
                }
            )

    scored_documents.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_documents[:top_k]


def print_results(query, results):
    """
    Print retrieval results.
    """

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    print(f"\nQuery: {query}")

    if not results:
        print("\nNo relevant documents found.")
        return

    for i, result in enumerate(
        results,
        start=1
    ):

        document = result["document"]
        score = result["score"]

        print(
            f"\n{i}. {document.get('title', 'Unknown')}"
        )

        print(
            f"   Category: "
            f"{document.get('category', 'Unknown')}"
        )

        print(
            f"   Score: {score}"
        )


if __name__ == "__main__":

    query = "How should I dispose plastic waste?"

    results = retrieve_documents(
        query,
        top_k=3
    )

    print_results(
        query,
        results
    )

    print("\nRetriever test completed.")