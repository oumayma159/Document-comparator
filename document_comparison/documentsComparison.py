import io
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from document_processing.documentPreprocessing import extract_text_from_pdf,preprocess_text
from document_processing.languageDetection import detect_language



def read_text_file(file):
    with io.open(file, 'r', encoding='utf-8') as f:
        return f.read()


def load_existing_documents(documents_dir = "data/organized_documents", ngram_range=(1, 2), max_features=10000):
    doc_paths = []
    contents = []
    languages = []

    for txt_file in Path(documents_dir).glob("*/texts/*.txt"):
        # print(f"Loading {txt_file}")
        content = read_text_file(str(txt_file))
        if not content.strip():
            continue
        lang = txt_file.parent.parent.name
        doc_paths.append(str(txt_file))
        contents.append(content)
        languages.append(lang)
    if contents:
        vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
        document_vectors = vectorizer.fit_transform(contents)
        documents_metadata = [
            {
                'path': path,
                'lang': lang,
                'content': content,
                'vector': document_vectors[i]
            }
            for i, (path, lang, content) in enumerate(zip(doc_paths, languages, contents))
        ]
        return vectorizer, document_vectors, documents_metadata
    else:
        print("No valid documents found")
        return None, None, []


def get_similarity(vectorizer, text1, text2):
    vec1 = vectorizer.transform(text1)
    vec2 = vectorizer.transform(text2)
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return similarity
    
    
def get_similarity_list(vectorizer, query_text, documents_metadata, lang_filter):
    if vectorizer is None:
        print("no vectorizer")
    query_vec = vectorizer.transform([query_text])
    results = []
    for doc in documents_metadata:
        if lang_filter and doc['lang'] != lang_filter:
            continue
        sim = cosine_similarity(query_vec, doc['vector'])[0][0]
        results.append({
            'similarity': float(sim),
            'document': doc
        })
    return sorted(results, key=lambda x: x['similarity'], reverse=True)


def compare_to_existing(pdf_path,documents_dir="data/organized_documents", ngram_range=(1, 2), max_features=10000):
    text = extract_text_from_pdf(pdf_path)
    processed_text = preprocess_text(text)       
    if not processed_text.strip():
        print("Empty text")          
    lang_code, lang_name = detect_language(processed_text)
    vectorizer, document_vectors, documents_metadata = load_existing_documents(
        documents_dir=documents_dir,
        ngram_range=ngram_range,
        max_features=max_features
    )
    # return similarity and doc (for every doc document data :path/lang/content/vector)
    similarities = get_similarity_list(vectorizer, processed_text, documents_metadata, lang_filter=lang_name)
    print(f"Found {len(similarities)} similar documents")
    matches = []
    for result in similarities:    
        if result['similarity'] > 0.85:
            status = 'duplicated document'
        elif 0.5 <= result['similarity'] <= 0.85:
            status = 'updated document'
        else:
            status = 'new document'
        matches.append({
            'path': result['document']['path'],
            'similarity': result['similarity'],
            'status': status
        })
    return {
        'metadata': {
            'source_path': pdf_path,
            'language': lang_name
        },
        'matches': matches,
    }
