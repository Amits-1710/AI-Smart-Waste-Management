import os
import json


DOCUMENTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "documents"
)


def load_documents():
    """
    Load all JSON knowledge-base documents.
    """

    documents = []

    if not os.path.exists(DOCUMENTS_DIR):
        return documents

    for filename in os.listdir(DOCUMENTS_DIR):

        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(DOCUMENTS_DIR, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                if isinstance(data, list):
                    documents.extend(data)

                elif isinstance(data, dict):
                    documents.append(data)

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    return documents


def get_all_documents():
    """
    Return all knowledge-base documents.
    """

    return load_documents()


def get_documents_by_category(category):
    """
    Return documents matching a category.
    """

    documents = get_all_documents()

    category = category.lower().strip()

    results = []

    for document in documents:

        document_category = str(
            document.get("category", "")
        ).lower().strip()

        if document_category == category:
            results.append(document)

    return results


def search_by_keyword(keyword):
    """
    Search documents using a keyword.
    """

    documents = get_all_documents()

    keyword = keyword.lower().strip()

    results = []

    for document in documents:

        title = str(
            document.get("title", "")
        ).lower()

        content = str(
            document.get("content", "")
        ).lower()

        keywords = document.get("keywords", [])

        keyword_text = " ".join(
            str(k).lower() for k in keywords
        )

        if (
            keyword in title
            or keyword in content
            or keyword in keyword_text
        ):
            results.append(document)

    return results


if __name__ == "__main__":

    print("=" * 60)
    print("AI SMART WASTE MANAGEMENT")
    print("KNOWLEDGE BASE TEST")
    print("=" * 60)

    documents = get_all_documents()

    print(f"\nTotal documents: {len(documents)}")

    print("\nAvailable documents:")

    for document in documents:
        print(
            f"- {document.get('title', 'Unknown')} "
            f"| Category: {document.get('category', 'Unknown')}"
        )

    print("\nKnowledge base test completed.")