import streamlit as st
import openai
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import time
from rank_bm25 import BM25Okapi

# Load FAISS index, BM25 index, and metadata
def load_faiss_index(index_path="faiss_index.index"):
    return faiss.read_index(index_path)

def load_bm25_index(index_path="bm25_index.pkl"):
    with open(index_path, 'rb') as f:
        return pickle.load(f)

def load_chunk_data(data_path="chunk.pkl"):
    with open(data_path, 'rb') as f:
        return pickle.load(f)

def load_chunk_metadata(metadata_path="chunk_metadata.pkl"):
    with open(metadata_path, 'rb') as f:
        return pickle.load(f)

# Retrieve relevant context using both FAISS and BM25
def get_relevant_context(query, faiss_index, bm25_index, chunk_data, chunk_metadata, model, k=3):
    query_vector = model.encode([query])
    D, I = faiss_index.search(query_vector, k)
    faiss_chunks = [chunk_metadata[i] for i in I[0]]
    
    tokenized_corpus = [doc.split() for doc in chunk_data]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_results = bm25.get_top_n(query.split(), chunk_data, n=k)
    
    relevant_chunks = list(set(faiss_chunks + bm25_results))
    return "\n\n".join(relevant_chunks), relevant_chunks[0]  

# Format response
def format_response(answer, source):
    return {"answer": answer, "source": source}

# OpenAI Chatbot function
openai.api_key = ""  # Replace with your actual API key

def ask_chatbot(question, context=None):
    messages = [{"role": "system", "content": "You are an insurance assistant helping with policy queries."}]
    if context:
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    else:
        messages.append({"role": "user", "content": question})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content

# Custom CSS for Web-style UI
st.markdown("""
    <style>
        .header {
            background-color: #075E54;
            color: white;
            text-align: center;
            padding: 15px;
            font-size: 24px;
            font-weight: bold;
            width: 100%;
        }
        .chat-container {
            width: 80%;
            margin: auto;
            padding: 20px;
        }
        .bot-message, .user-message {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .bot-text {
            background-color: #dcf8c6;
            color: black;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
        }
        .user-text {
            background-color: #34B7F1;
            color: white;
            padding: 10px;
            border-radius: 10px;
            max-width: 70%;
            align-self: flex-end;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">💬 RAG-Enhanced Insurance Assistant</div>', unsafe_allow_html=True)

# Initialize session state for conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = [("Bot", "Hello! How may I help you today?")]

if "awaiting_query" not in st.session_state:
    st.session_state.awaiting_query = False

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, text in st.session_state.conversation:
    if role == "Bot":
        st.markdown(f"""
            <div class="bot-message">
                <div class="bot-text">{text}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="user-message" style="justify-content: flex-end;">
                <div class="user-text">{text}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")

if not st.session_state.awaiting_query:
    with st.form("query_form"):
        user_input = st.text_input("Enter your question here...")
        submit_button = st.form_submit_button("Submit")

    if submit_button and user_input:
        with st.spinner("Processing your query..."):
            faiss_index = load_faiss_index()
            bm25_index = load_bm25_index()
            chunk_data = load_chunk_data()
            chunk_metadata = load_chunk_metadata()
            model = SentenceTransformer("all-MiniLM-L6-v2")
            context, source = get_relevant_context(user_input, faiss_index, bm25_index, chunk_data, chunk_metadata, model)
            answer = ask_chatbot(user_input, context)
            response = format_response(answer, source)

        st.session_state.conversation.append(("You", user_input))
        st.session_state.conversation.append(("Bot", response["answer"]))
        st.session_state.awaiting_query = True
        st.experimental_rerun()

if st.session_state.awaiting_query:
    query_more = st.radio("Any further queries?", ["Yes", "No"], index=None)
    if query_more == "Yes":
        st.session_state.awaiting_query = False
        st.experimental_rerun()
    elif query_more == "No":
        feedback = st.radio("Was this helpful?", ["✔ Yes", "✖ No"], index=None)
        if feedback:
            if feedback == "✔ Yes":
                st.success("Thank you for your feedback!")
            else:
                st.warning("We will try better!")
            time.sleep(2)
            st.session_state.conversation = [("Bot", "Hello! How may I help you today?")]
            st.session_state.awaiting_query = False
            st.experimental_rerun()
