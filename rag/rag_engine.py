import os
from dotenv import load_dotenv
from langdetect import detect
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from rag.preprocessing import build_vector_store

load_dotenv()


TEXT_PATH = os.getenv("TEXT_PATH", "data/HSC26-Bangla1st-Paper.pdf")
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "vector_store")

llm = OllamaLLM(model="gemma3")

embedding_model = HuggingFaceEmbeddings(
    model_name="l3cube-pune/bengali-sentence-similarity-sbert",
    model_kwargs={"device": "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"}
)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


EN_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are a smart multilingual assistant. Use the context to answer clearly.

Question:
{question}

Context:
{context}

Answer:"""
)

BN_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""তুমি একজন বুদ্ধিমান সহকারী। নিচের প্রসঙ্গ ব্যবহার করে স্পষ্টভাবে প্রশ্নের উত্তর দাও।

প্রশ্ন:
{question}

প্রসঙ্গ:
{context}

উত্তর:"""
)

def get_prompt_template(question: str):
    try:
        lang = detect(question)
        return BN_PROMPT if lang == "bn" else EN_PROMPT
    except Exception:
        return EN_PROMPT  


def get_rag_chain(persist_dir=VECTOR_STORE_DIR):
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        build_vector_store(TEXT_PATH, persist_dir)
    else:
        print("Using existing vector store.")

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )
    

    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    def chain_factory(question: str):
        prompt = get_prompt_template(question)
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
    
    return chain_factory
