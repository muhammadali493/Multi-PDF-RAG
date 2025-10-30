from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def build_rag_chain(retriever, prompt, llm):
    """
    Build a RAG chain using LangChain's built-in chains.
    This handles document formatting and context passing automatically.
    """
    # Create a history-aware retriever that reformulates questions based on chat history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, "
                   "formulate a standalone question which can be understood without the chat history. "
                   "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    print(f'History aware retriever is of type: {type(history_aware_retriever)}')
    # Create the question-answering chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Combine retrieval and QA into a single chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain