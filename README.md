# Multi-PDF RAG Application 📄

An intelligent **document Q&A system** that enables **conversational retrieval across multiple PDF documents** using **LangChain** and **ChromaDB**.

Ask questions about your documents and receive **context-aware, cited responses** powered by **OpenAI's GPT-4o-mini** and advanced retrieval techniques.

## 🌟 Features

- **Multi-Document Intelligence:** Upload and query multiple PDFs simultaneously.  
- **Smart Memory Management:** Dynamic chat history summarization reduces token usage by **60%** while preserving conversation context.  
- **Parallel Processing:** `ThreadPoolExecutor`-based ingestion pipeline reduces batch upload processing time by **75%**.  
- **Intelligent Deduplication:** SHA-256 hash-based system prevents redundant indexing of identical documents.  
- **Multi-Query Retrieval:** Automatically generates query paraphrases to improve retrieval accuracy and recall.  
- **Thread-Safe Architecture:** Concurrent file processing without race conditions or data corruption.  
- **Persistent Storage:** ChromaDB vector store with automatic session state management.
  


