# filepath: d:\ML Projects\Langchain\rag_app\chains\rag_chain.py
from typing import List
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
#from rag_app.retrieval.multi_query import multi_query_retrieve  # new multi-query helper
from rag_app.settings import settings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain, LLMChain

def join_context(docs: List[Document]) -> str:
    return "\n".join(
        "Source: " + str(d.metadata.get("source", "Unknown")) +
        (f"\nPage No: {d.metadata.get('page')}" if 'page' in d.metadata else "") +
        "\n" + d.page_content
        for d in docs
    )



def build_rag_chain(retriever, prompt, llm):
    return (
        #{
        #    "context": retriever | RunnableLambda(join_context),
        #    "chat_history": RunnablePassthrough(),  # pass-through from input
        #    "question": RunnablePassthrough(),      # pass-through from input
        #}
        {
            "context": itemgetter("question") | retriever | RunnableLambda(join_context),
            "chat_history": itemgetter("chat_history"),  # extract chat_history field
            "question": itemgetter("question"),          # extract question field
        }
        | prompt
        | llm
        | StrOutputParser()
    )
"""
def build_rag_chain(retriever, prompt, llm, use_multi_query: bool = True, n_queries: int = 4):
    
    Build a RAG pipeline. By default, use multi-query retrieval:
      - itemgetter("question") feeds into a RunnableLambda that runs multi_query_retrieve(...)
      - results are then joined into context via RunnableLambda(join_context)
    If use_multi_query is False, the provided retriever is used directly (as before).
    
    if use_multi_query:
        # Create a lambda that accepts a question and returns Documents (unique set)
        def _mq_wrapper(question):
            # retriever: base retriever (e.g. Chroma retriever)
            docs = multi_query_retrieve(
                base_retriever=retriever,
                llm=llm,
                question=question,
                n_queries=n_queries,
                k_per_query=settings.top_k,
                where=None
            )
            return docs

        context_runnable = itemgetter("question") | RunnableLambda(_mq_wrapper) | RunnableLambda(join_context)
    else:
        # use original single-query retriever pipeline
        context_runnable = itemgetter("question") | retriever | RunnableLambda(join_context)

    return (
        {
            "context": context_runnable,
            "chat_history": itemgetter("chat_history"),  # extract chat_history field
            "question": itemgetter("question"),          # extract question field
        }
        | prompt
        | llm
        | StrOutputParser()
    )
"""