from rag.preprocess import extract_text_from_pdf, clean_and_chunk_text

if __name__ == "__main__":
    path = "data/HSC26-Bangla1st-Paper.pdf"
    text = extract_text_from_pdf(path)
    print("✅ Extracted text length:", len(text))

    chunks = clean_and_chunk_text(text)
    print("✅ Total chunks created:", len(chunks))

    print("\n🔹 Sample Chunk:\n")
    print(chunks[0][:1000])  # Show first 1000 characters of first chunk
