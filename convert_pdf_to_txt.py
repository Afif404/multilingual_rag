from pdfminer.high_level import extract_text

pdf_path = "data/HSC26-Bangla1st-Paper.pdf"
txt_path = "data/HSC26-Bangla1st-Paper.txt"

text = extract_text(pdf_path)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(text)

print("✅ Converted PDF to TXT")
