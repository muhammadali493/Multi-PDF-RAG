import os, tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

from rag_app.settings import settings
from rag_app.models.llm import create_llm
from rag_app.models.embeddings import create_embeddings
from rag_app.utils.hashing import sha256_bytes
from rag_app.ingestion.pdf_loader import load_pdf
from rag_app.ingestion.chunker import split_documents
from rag_app.ingestion.pipeline import enrich_metadata
from rag_app.vectorstore.chroma_store import ChromaStore
from rag_app.vectorstore.processed_repo import ProcessedHashesRepo
from rag_app.retrieval.retriever import build_source_filter
from rag_app.prompts.qa import qa_prompt
from rag_app.chains.rag_chain import build_rag_chain

load_dotenv()
st.set_page_config(page_title='ChatPDF', layout='wide')
st.title('Welcome to ChatPDFs')

# Thread-safe lock for updating shared state
state_lock = Lock()

@st.cache_resource
def get_embeddings():
    return create_embeddings()

@st.cache_resource
def get_llm():
    return create_llm()

@st.cache_resource
def get_vector_store():
    return ChromaStore(embeddings=get_embeddings(), persist_dir=settings.chroma_dir)

@st.cache_resource
def get_processed_repo():
    return ProcessedHashesRepo(os.path.join(settings.chroma_dir, "processed_hashes.json"))

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'file_list' not in st.session_state:
    st.session_state.file_list = []
if 'processed_hashes' not in st.session_state:
    st.session_state.processed_hashes = get_processed_repo().load()

vector_store = get_vector_store()
llm = get_llm()

st.markdown("<div style='height:30vh;'></div>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload your PDF files", type=['pdf'], accept_multiple_files=True)

def process_single_file(file_data, processed_hashes_snapshot, vector_store):
    """
    Process a single PDF file. Returns a dict with status info.
    This function is designed to be thread-safe by NOT accessing st.session_state.
    
    Args:
        file_data: Tuple of (filename, file_bytes)
        processed_hashes_snapshot: Set of already processed file hashes (read-only)
        vector_store: The vector store instance (thread-safe)
    """
    name, data = file_data
    file_hash = sha256_bytes(data)
    
    # Check if already processed (using passed-in snapshot, no session state access)
    if file_hash in processed_hashes_snapshot:
        return {
            'name': name,
            'status': 'skipped',
            'message': f"Skipping {name}: already indexed.",
            'file_hash': file_hash
        }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        path = tmp.name
    
    try:
        # Load PDF pages
        pages = load_pdf(path)
        
        # Split into chunks
        chunks = split_documents(pages)
        
        # Enrich metadata
        chunks = enrich_metadata(chunks, filename=name, file_hash=file_hash)
        
        # Add to vector store (ChromaDB is thread-safe)
        added_ids = vector_store.add_documents(chunks)
        
        return {
            'name': name,
            'status': 'success',
            'message': f"Added {len(added_ids)} chunks from {name} ({len(pages)} pages).",
            'file_hash': file_hash,
            'num_pages': len(pages),
            'num_chunks': len(added_ids)
        }
        
    except Exception as e:
        return {
            'name': name,
            'status': 'error',
            'message': f"Error processing {name}: {str(e)}",
            'file_hash': file_hash
        }
    finally:
        # Clean up temporary file
        try:
            os.remove(path)
        except OSError:
            pass

# Parallel ingestion pipeline
if uploaded_files:
    start_time = time.time()
    st.info(f"Processing {len(uploaded_files)} files in parallel...")
    
    # Prepare file data (read all files in main thread)
    file_data_list = [(uf.name, uf.read()) for uf in uploaded_files]
    
    # Create a snapshot of processed hashes to pass to workers (avoiding session state access)
    processed_hashes_snapshot = st.session_state.processed_hashes.copy()
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process files in parallel
    results = []
    max_workers = min(4, len(file_data_list))  # Limit to 4 parallel workers
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks (pass snapshot and vector_store to each worker)
        future_to_file = {
            executor.submit(process_single_file, fd, processed_hashes_snapshot, vector_store): fd[0] 
            for fd in file_data_list
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)
            completed += 1
            
            # Update progress
            progress = completed / len(file_data_list)
            progress_bar.progress(progress)
            status_text.text(f"Processed {completed}/{len(file_data_list)} files...")
            
            # Show toast for each result
            st.toast(result['message'])
    
    # Update session state ONLY in main thread (thread-safe batch update)
    for result in results:
        if result['status'] in ('success', 'skipped'):
            # Add to processed hashes
            st.session_state.processed_hashes.add(result['file_hash'])
            
            # Add to file list if not already there
            if result['name'] not in st.session_state.file_list:
                st.session_state.file_list.append(result['name'])
    
    # Save processed hashes to disk
    get_processed_repo().save(st.session_state.processed_hashes)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display summary
    end_time = time.time()
    elapsed = end_time - start_time
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')
    
    st.toast(f"""
    Processing complete in {elapsed:.2f} seconds!
    - Processed: {success_count}
    - Skipped: {skipped_count}
    - Errors: {error_count}
    """)
    
    # Show detailed results
    if error_count > 0:
        with st.expander("⚠️ View Errors"):
            for result in results:
                if result['status'] == 'error':
                    st.error(result['message'])

# Source selection
file_options = ["All files"] + st.session_state.file_list
selected = st.multiselect("Select files to query", options=file_options)

# Display history
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

user_query = st.chat_input("Ask your question...")
if not st.session_state.file_list and not uploaded_files:
    st.info("Please upload at least one PDF document first.")
elif user_query:
    if not selected:
        st.warning("Please select at least one file from drop down to ask question.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.write(user_query)

        where = build_source_filter(selected, st.session_state.file_list) or {"source": {"$in": st.session_state.file_list}}
        retriever = vector_store.as_retriever(k=settings.top_k, where=where)
        
        # Keep only last 5 previous messages (excluding current)
        recent_history = [m for m in st.session_state.chat_history[:-1]][-5:]
        print(f'Recent history: {recent_history}')
        rag = build_rag_chain(retriever=retriever, prompt=qa_prompt, llm=llm)
        
        # Built-in chains return dict with 'answer' key and expect 'input' instead of 'question'
        response = rag.invoke({"input": user_query, "chat_history": recent_history})
        print(response)
        answer = response["answer"]
        
        st.session_state.chat_history.append(AIMessage(content=answer))
        with st.chat_message("assistant"):
            st.write(answer)
else:
    st.info("Please enter a question in the chat box.")