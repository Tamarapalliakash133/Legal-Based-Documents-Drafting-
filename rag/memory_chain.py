from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_openai import ChatOpenAI
from langsmith import traceable

import rag.config as config
from rag.retriever import build_retriever
import rag.config as config

PDF_QA_SYSTEM_PROMPT = """
You are a legal assistant answering questions using the provided PDF documents.

Use the retrieved context as the source of information for document-specific answers.
Do not invent legal provisions, facts, or citations.

If the retrieved context does not contain enough information, clearly explain
that the answer could not be verified from the provided documents.

Retrieved context:
{context}
"""

GENERAL_SYSTEM_PROMPT = """
You are a helpful general-purpose AI assistant.

The retrieved PDF context is empty, so answer general questions using your
general knowledge.

For questions about specific legal provisions or the supplied documents,
clearly state when the relevant information could not be found.
"""


def get_session_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url=config.REDIS_URL,
    )


def build_converstional_rag_chain():

    llm = ChatOpenAI(model= config.LLM_MODEL)
    
    retriever = build_retriever(llm)

    pdf_prompt = ChatPromptTemplate.from_messages([
        ("system", PDF_QA_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    pdf_chain = create_stuff_documents_chain(
        llm,
        pdf_prompt,
    )

    general_prompt = ChatPromptTemplate.from_messages([
        ("system", GENERAL_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    general_chain = (
        general_prompt
        | llm
        | RunnableLambda(lambda message: message.content)
    )


    answer_chain = RunnableBranch(
        (
            lambda inputs: not inputs.get("context"),
            general_chain,
        ),
        pdf_chain,
    )

    rag_chain = create_retrieval_chain(
        retriever,
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