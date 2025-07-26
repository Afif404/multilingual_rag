import os
import re
from typing import List
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

os.environ["CHROMA_TELEMETRY_ENABLED"] = "FALSE"

embedding_model = HuggingFaceEmbeddings(
    model_name="l3cube-pune/bengali-sentence-similarity-sbert",
    model_kwargs={"device": "cpu"}
)


def bijoy_to_unicode(text: str) -> str:
    mapping = {
        'Avwg': 'আমি',
        '‡Kvb': 'কাজ',
        'evsjv': 'বাংলা',
        '†K›`ª': 'প্রশ্ন',
        'g‡b': 'উত্তর',
        'mv‡_': 'সময়',
        '`vI': 'হয়',
        'bv‡g': 'পরে',
        'cvZ': 'জীবন',
        # ➕ Add more mappings as needed (from Bijoy layout chart)
    }
    for ascii_bijoy, uni in mapping.items():
        text = text.replace(ascii_bijoy, uni)
    return text


def extract_text_from_pdf(path: str) -> str:
    print(f"Extracting from PDF: {path}")
    try:
        raw_text = extract_text(path)
        return bijoy_to_unicode(raw_text)
    except Exception as e:
        print(f"Failed to extract text: {e}")
        return ""

def clean_text(text: str) -> str:
    text = text.replace('\xa0', ' ')
    text = re.sub(r'[^\u0980-\u09FF0-9A-Za-z\s\.\,\?\!:\-–—\n]', '', text)
    lines = text.split("\n")
    return "\n".join([line.strip() for line in lines if line.strip()])

def split_mcq_blocks(text: str) -> List[str]:
    return [b.strip() for b in re.split(r"(?:প্রশ্ন\s*\d+|^\d+\.)", text, flags=re.MULTILINE) if len(b.strip()) > 40]

def split_story_blocks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["।", "\n", "?", "!", "."]
    )
    return splitter.split_text(text)

def detect_mcq_section(text: str) -> str:
    if "প্রশ্ন" in text and "উত্তর" in text:
        return "MCQ"
    return "STORY"

def chunk_text(text: str) -> List[str]:
    mcqs, stories = [], []
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        kind = detect_mcq_section(para)
        if kind == "MCQ":
            mcqs.extend(split_mcq_blocks(para))
        else:
            stories.extend(split_story_blocks(para))
    return [c for c in mcqs + stories if len(c.strip()) > 30]

def build_vector_store(text_path: str, persist_dir: str = "vector_store"):
    print(" Building vector store...")
    raw = extract_text_from_pdf(text_path)
    clean = clean_text(raw)
    chunks = chunk_text(clean)
    print(f"Total Chunks: {len(chunks)}")

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print(f"Vector store saved to: {persist_dir}")
    return vectordb
