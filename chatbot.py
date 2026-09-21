from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embedding model loaded successfully.")

print("Loading vector database...")

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

print("Vector database loaded successfully.")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

print("Loading local LLM...")

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)

print("LLM loaded successfully.")

print()
print("===================================")
print("          RAG Chatbot Ready")
print("===================================")
print("Ask questions about your documents.")
print("Type 'exit' to quit.")
print()

while True:

    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    try:

        documents = retriever.invoke(question)

        if not documents:
            print("\nBot: I don't know based on the provided documents.\n")
            continue

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are a helpful document-based RAG chatbot.

Answer the user's question using ONLY the information
provided in the context below.

Rules:
- Do not use outside knowledge.
- Do not make up information.
- If the answer is not present in the context, say:
  "I don't know based on the provided documents."
- Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        print("\nBot:", response.content)

        print("\nRetrieved sources:")

        for i, document in enumerate(documents, start=1):

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            print(f"{i}. {source}")

        print()

    except Exception as e:

        print("\nError:", e)
        print()
