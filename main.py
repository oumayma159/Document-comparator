import sys
import os
from document_processing.languageDetection import organize_all_documents
from document_comparison.documentsComparison import compare_to_existing


def main(pdf_path):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    base_folder = os.path.join(BASE_DIR, "data", "base_documents")
    output_dir = os.path.join(BASE_DIR, "data", "organized_documents")

    doc_info = organize_all_documents(base_folder, output_dir=output_dir)

    if doc_info:
        print(f"Loading existing documents under language: {doc_info[0]['lang_name']}")

    print("Comparing...")

    results = compare_to_existing(
        pdf_path, documents_dir=output_dir, ngram_range=(1, 2), max_features=10000
    )

    if not results:
        print("Could not extract text from the document.")

        return

    metadata = results["metadata"]
    matches = results["matches"]

    print(f"Document language: {metadata['language']}")

    if not matches:
        print("Could not find any similar documents.")

    relevant_matches = matches[:5]

    print(
        f"\nComparison result: {metadata['status']} with the best similarity being {metadata['best_similarity']:.2f}"
    )

    for match in relevant_matches:
        print(
            f"- the document stored in {match['path']} has a similarity of {match['similarity']:.2f} with the new document"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("please execute: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    main(pdf_path)
