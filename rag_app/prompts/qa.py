from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI Assistant. Use the provided context to answer the user's question.
                    If the answer cannot be found in the context, say you don't know. Don't provide information without context relevant to the query.
                    Context:
                    {context}
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
