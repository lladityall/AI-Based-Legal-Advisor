import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client
client = chromadb.Client()

# Create or load a collection for legal documents
collection = client.get_or_create_collection(name="legal_knowledge")

# Load a pre-trained embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Add legal documents to the knowledge base (example data)
legal_documents = [
    {"id": "1", "text": "A valid contract requires an offer, acceptance, consideration, and intention to create legal relations."},
    {"id": "2", "text": "Criminal liability requires both actus reus (guilty act) and mens rea (guilty mind)."},
    {"id": "3", "text": "The penalty for breach of contract may include damages, specific performance, or injunctions."},
    {"id": "4", "text": "In tort law, negligence requires a duty of care, breach of duty, causation, and damages."},
]

# Embed and store legal documents in ChromaDB
for doc in legal_documents:
    embedding = embedding_model.encode(doc["text"])
    collection.add(
        ids=[doc["id"]],
        embeddings=[embedding.tolist()],
        documents=[doc["text"]]
    )

# Retrieve relevant legal information
def retrieve_legal_info(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]

# Generate response using Ollama DeepSeek model
def generate_response(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = ollama.generate(model="deepseek-r1:8b", prompt=prompt)
    return response["response"]

# Streamlit app
def main():
    st.title("AI-Powered Legal Advisor")
    st.write("Welcome! How can I assist you with your legal queries today?")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask your legal question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Step 1: Retrieve relevant legal information
        context = retrieve_legal_info(prompt)
        context_str = "\n".join(context)

        # Step 2: Generate response using Ollama DeepSeek
        response = generate_response(prompt, context_str)

        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
