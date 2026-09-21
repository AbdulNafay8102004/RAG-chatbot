import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

TXT_PATH = "documents/handbook.txt"
DB_PATH = "chroma_db"

print("Loading document...")

loader = TextLoader(
TXT_PATH,
encoding="utf-8"
)

documents = loader.load()

print(f"Loaded document successfully")

print("Splitting document...")

text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1000,
chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating vector database...")

vectorstore = Chroma.from_documents(
documents=chunks,
embedding=embeddings,
persist_directory=DB_PATH
)

print("Vector database created successfully!")
