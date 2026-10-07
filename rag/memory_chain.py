from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_openai import ChatOpenAI

import rag.config as config 
from rag.retriever import build_retriever
from store_logs.db import collection

CONTEXTUALIZE_SYSTEM_PROMPT = """
Given the chat history and the latest user question, determine whether the
latest question depends on something mentioned earlier in the conversation.

If it does, rewrite the question into a standalone question that can be
understood without the chat history.

Do NOT answer the question.

Only reformulate the question when necessary.

If the question is already standalone, return it unchanged.
"""

PDF_QA_SYSTEM_PROMPT = """
You are an assistant answering questions using a specific PDF document.

Use ONLY the retrieved PDF context below to answer the user's question.

Do not use outside knowledge when answering from the PDF.

If the retrieved context does not contain enough information to answer the
question, you can answer with the llm.

Retrieved PDF context:

{context}
"""

GENERAL_SYSTEM_PROMPT = """
You are a helpful general-purpose AI assistant.

The user's question could not be answered using the PDF retrieval system.

Answer the user's question using your general knowledge.

Do not claim that the answer came from the PDF.

If the question refers specifically to information that should exist inside
the PDF but you do not have that information, clearly tell the user that the
information was not found in the document.
"""


def get_session_history(session_id) -> RedisChatMessageHistory:

    return RedisChatMessageHistory(
        session_id = session_id,
        url = config.REDIS_URL,
    )


def build_converstional_rag_chain():

    llm = ChatOpenAI(model = "gpt-4o-mini")

    retriever = build_retriever(llm)

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human","{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt
    )

    pdf_prompt = ChatPromptTemplate.from_messages([
        ("system",PDF_QA_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human","{input}"),
    ])

    pdf_document_chain = create_stuff_documents_chain(
        llm,
        pdf_prompt
    )


    general_prompt = ChatPromptTemplate.from_messages([
        ("system",GENERAL_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human","{input}"),
    ])

    general_chain = general_prompt | llm

    general_answer_chain = (
        general_chain |
        RunnableLambda(lambda message : message.content)
    )

    pdf_answer_chain = pdf_document_chain

    answer_chain = RunnableBranch(
    (
        lambda inputs: len(inputs.get("context", [])) == 0,
        general_answer_chain
    ),
    pdf_answer_chain
)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        answer_chain,
    )

    
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain