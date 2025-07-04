import os
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini 1.5 Flash
model = genai.GenerativeModel("gemini-1.5-flash")

# Load PDF and extract text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

text = extract_text_from_pdf("surah1.pdf")

# Split into chunks
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = split_text(text)

# Setup Chroma DB with fake embedding (to simulate retrieval)
embedding_function = DefaultEmbeddingFunction()
chroma = Client(Settings(anonymized_telemetry=False))
collection = chroma.create_collection(name="surah_chunks", embedding_function=embedding_function)

# Add chunks to collection
for i, chunk in enumerate(chunks):
    collection.add(documents=[chunk], ids=[str(i)])

# QA Loop
while True:
    query = input("\nAsk something about the Surah (type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    results = collection.query(query_texts=[query], n_results=3)
    relevant_docs = "\n".join([doc for doc in results['documents'][0]])

    prompt = f"Based on the following context from the Surah, answer this:\n\nContext:\n{relevant_docs}\n\nQuestion: {query}"

    response = model.generate_content(prompt)
    print("\nAnswer:", response.text.strip())
