import io
from pathlib import Path
from document_processing.documentPreprocessing import (
    extract_text_from_pdf,
    preprocess_text,
)
from document_processing.languageDetection import detect_language
from sentence_transformers import SentenceTransformer, util


def read_text_file(file):
    with io.open(file, "r", encoding="utf-8") as f:
        return f.read()


def load_existing_documents(
    documents_dir="data/organized_documents",
    lang_filter=None,
    ngram_range=(1, 2),
    max_features=10000,
):
    doc_paths = []
    contents = []
    languages = []

    glob_pattern = "*/texts/*.txt"

    if lang_filter:
        glob_pattern = f"{lang_filter}/texts/*.txt"

    for txt_file in Path(documents_dir).glob(glob_pattern):
        content = read_text_file(str(txt_file))

        if not content.strip():
            continue

        lang = txt_file.parent.parent.name

        doc_paths.append(str(txt_file))
        contents.append(content)
        languages.append(lang)

    if not contents:
        print("No valid documents found")

        return None, None, []

    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    document_vectors = model.encode(contents, convert_to_tensor=True)
    documents_metadata = [
        {"path": path, "lang": lang, "content": content, "vector": vector}
        for path, lang, content, vector in zip(
            doc_paths, languages, contents, document_vectors
        )
    ]

    return model, document_vectors, documents_metadata


def get_similarity_list(model, query_text, documents_metadata):
    if model is None:
        print("No model available")

        return []

    query_vec = model.encode(query_text, convert_to_tensor=True)
    results = []

    for doc in documents_metadata:
        sim = util.cos_sim(query_vec, doc["vector"]).item()
        results.append({"similarity": float(sim), "document": doc})

    return sorted(results, key=lambda x: x["similarity"], reverse=True)


def compare_to_existing(
    pdf_path,
    documents_dir="data/organized_documents",
    ngram_range=(1, 2),
    max_features=10000,
):
    text = extract_text_from_pdf(pdf_path)
    processed_text = preprocess_text(text)

    if not processed_text.strip():
        print("Empty text")

        return None

    lang_code, lang_name = detect_language(processed_text)
    model, document_vectors, documents_metadata = load_existing_documents(
        documents_dir=documents_dir,
        lang_filter=lang_name,
        ngram_range=ngram_range,
        max_features=max_features,
    )

    # return similarity and doc (for every doc document data :path/lang/content/vector)
    similarities = get_similarity_list(model, processed_text, documents_metadata)
    print(f"Found {len(similarities)} similar document(s).")

    matches = [
        {
            "path": result["document"]["path"],
            "similarity": result["similarity"],
        }
        for result in similarities
    ]

    metadata = {"source_path": pdf_path, "language": lang_name}

    if matches:
        best_similarity = matches[0]["similarity"]

        if best_similarity >= 0.99:
            status = "Duplicated document"
        elif best_similarity >= 0.7:
            status = "Updated document"
        else:
            status = "New document"

        metadata["best_similarity"] = best_similarity
        metadata["status"] = status

    return {
        "metadata": metadata,
        "matches": matches,
    }
