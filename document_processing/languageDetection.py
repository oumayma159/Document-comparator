from langdetect import detect
from langcodes import Language
from document_processing.documentPreprocessing import extract_text_from_pdf,preprocess_text
import shutil
from pathlib import Path


def detect_language(text, fallback='en'):
    if not text.strip():
        return fallback, Language.get(fallback).display_name('en')     
    try:
        sample_text = text[:1000] if len(text) > 1000 else text
        language_code = detect(sample_text) 
        language_name = Language.get(language_code).display_name('en')

    except Exception as e:
        print(f"Language detection failed: {str(e)}")
        language_code = fallback
        language_name = Language.get(fallback).display_name('en')
    
    return language_code, language_name

# def filter_documents_by_language(pdf_path,output_dir="filtered_documents",target_language):
#     text = extract_text_from_pdf(pdf_path)
#     preprocessed_text = preprocess_text(text)
#     language_code, language_name = detect_language(preprocessed_text)
#     if (language_code == target_language.lower()):
#         lang_dir = Path(output_dir) / language_name
#         lang_dir.mkdir(parents=True, exist_ok=True)
        
def organize_document(pdf_path, output_dir):
    text = extract_text_from_pdf(pdf_path) 
    preprocessed_text = preprocess_text(text) 
    lang_code, lang_name = detect_language(preprocessed_text)
    
    pdf_dir = Path(output_dir) / lang_name / "pdfs"
    txt_dir = Path(output_dir) / lang_name / "texts"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    
    filename = Path(pdf_path).stem
    shutil.copy2(pdf_path, pdf_dir / f"{filename}.pdf")
    with open(txt_dir / f"{filename}.txt", 'w', encoding='utf-8') as f:
        f.write(preprocessed_text)
    return {
            'pdf_path': pdf_path,
            'lang_code': lang_code,
            'lang_name': lang_name
        }
    
def organize_all_documents(base_folder, output_dir):
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_folder)
    pdf_files = list(base_path.glob("*.pdf"))
    results = []
    if not pdf_files:
        print("No PDF documents found in the base folder.")
        return
    for pdf_path in pdf_files:
        try:
            doc_info = organize_document(str(pdf_path), output_dir=output_dir)
            # print(f"✅ Document '{pdf_path.name}' organized under language: {doc_info['lang_name']}")
            results.append(doc_info)
        except Exception as e:
            print(f"Error processing ")
    print(results)
    
   
    
   