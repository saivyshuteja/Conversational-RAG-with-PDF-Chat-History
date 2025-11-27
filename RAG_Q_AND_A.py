import streamlit as st
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

st.title("Conversational Rag with pdf uploads and chat history")
st.write("Upload pdf's and chat with their content")
api_key=st.text_input("enter your groq api key ",type="password")
if api_key:
    llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
             # optional
    )
        
    session_id = st.text_input("session_id", value="Default_session")
    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files=st.file_uploader("choose a pdf file ",type="pdf",accept_multiple_files=True)
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vector_store=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vector_store.as_retriever()
        contextualize_q_system_prompt=(
            "given a chat history and the latest user question"
            "which might referece content i the chat history"
            "formulate a standalone question which ca be understand"
            "without the chat history do not answer the question"
            "just reformulate it if needed an otherwise return it s it is "
        )
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )
        history_aware_retriever=create_history_aware_retriever(llm,retriever, contextualize_q_prompt)
        system_prompt=(
            "you are a assistant for uestion aswering tasks"
            "use the following pieces of retrieved context to answer"
            "the uestio if you dont know the answer, say with you"
            "dont know use three sentences maximum and keep the "
            "answer concise"
            "\n\n"
            "{context}"
        )
        qa_prompt= ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),

            ]
        )
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(  history_aware_retriever, question_answer_chain)
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        converstional_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        user_input=st.text_input("your question")
        if user_input:
            session_history=get_session_history(session_id)
            response=converstional_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
                
            )
            st.write(st.session_state.store)
            st.success(f"assistant: {response['answer']}")
            st.write("chat history: ",session_history.messages)
else:
    st.warning("please enter correct the groq_api_key")


    