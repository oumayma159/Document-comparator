import pdfplumber
import fitz
import pytesseract
from PIL import Image
from io import BytesIO
import re

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"   
        if not text.strip():
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text:
                        text += page_text + "\n"    
        if not text.strip():
            text = extract_text_with_ocr(pdf_path)         
    except Exception as e:
        print(f"Failed to extract text : {str(e)}")
    
    return text.strip()

def pdf_to_images(pdf_page, dpi=300):
    zoom_x = dpi / 72.0
    zoom_y = dpi / 72.0
    mat = fitz.Matrix(zoom_x, zoom_y)
    pix = pdf_page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("ppm")
    pil_img = Image.open(BytesIO(img_bytes))
    return pil_img

def extract_text_with_ocr(pdf_path):
    total_text = ""
    with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                image = pdf_to_images(page)
                if image:          
                    custom_config = r'--oem 3 --psm 6'
                    ocr_text = pytesseract.image_to_string(image, lang='eng+deu', config=custom_config)
                    total_text += ocr_text + "\n"
    return total_text.strip()


def preprocess_text(text):
    if not text:
        return ""
    text = re.sub(r'^\s*.*(?:page|seite|document|dokument)\s*\d+.*$', '', text, flags=re.MULTILINE|re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    text = text.lower()
    
    return text
