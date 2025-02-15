(Priyanka- k67900): Data extraction—pre-processing 

Extracted text and Pre-Processed Policy Documents
Cleaned and structured policy content into chunks.
Implemented FAISS-Based Semantic Search
Generated vector embeddings for document chunks.
Indexed embeddings using FAISS for fast retrieval.
Implemented query-to-vector search using FAISS.
Integrated FAISS with GPT-3.5

(Bilal - k67963): BM25, Hybrid Ranking & Query Expansion
Implemented BM25 for Lexical Search
Tokenized and indexed document chunks.
Developed Hybrid FAISS + BM25 Ranking
Combined FAISS semantic similarity with BM25 lexical search.
Applied a weighted ranking formula for better retrieval accuracy.
Implemented Query Expansion Using GPT-3.5
Optimized Retrieval Pipeline & Debugging
Evaluated and fine-tuned hybrid search performance.

(Anusha - 67910) : The Streamlit-based RAG chatbot features a WhatsApp Web-style UI, integrating FAISS (semantic search) and BM25 (keyword search) for retrieving relevant text chunks. User queries are processed through both methods, providing context to OpenAI's GPT-3.5 for accurate responses.

The chatbot maintains a dynamic conversation flow, prompting "Any further queries?" with Yes/No buttons. Selecting "Yes" allows new queries, while "No" triggers a feedback prompt before resetting the chat. Session state ensures seamless interaction, preserving chat history without page refreshes. This implementation delivers an interactive and efficient AI assistant using Streamlit’s real-time UI capabilities.
