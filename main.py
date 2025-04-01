import sys
from document_processing.languageDetection import organize_all_documents
from document_comparison.documentsComparison import load_existing_documents, compare_to_existing

def main(pdf_path):
    output_dir = "data/organized_documents"
    base_folder = "C:/Users/dell/Desktop/TECH/ML-AI/LLMs/DocumentComparator/data/base_documents"
    doc_info = organize_all_documents(base_folder, output_dir=output_dir)
    if doc_info:
        print(f"Loading existing documents uder language: {doc_info['lang_name']}")
    print("Comparing")
    results = compare_to_existing(pdf_path,documents_dir=output_dir, ngram_range=(1, 2), max_features=10000)
    print("\nComparison Results:")
    print(f"Source Document Language: {results['metadata']['language']}")
    for match in results['matches']:
        print(f"- {match['status']}: the document stored in {match['path']} has a similarity of {match['similarity']:.2f} with the new document")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("please execute: python main.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    main(pdf_path)
