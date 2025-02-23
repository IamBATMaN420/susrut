import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🌟 AI Chatbot")
    st.markdown("""
    - Uses **Mistral-7B-Instruct**
    - Provides accurate responses from context
    - No hallucination mode
    """)
    st.markdown("---")
    st.button("Clear Chat", on_click=lambda: st.session_state.pop("messages", None))

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def main():
    st.markdown("""
    <style>
    .stChatMessage.user {background-color: #2a2a2a; color: #ffffff; border-radius: 10px; padding: 10px; margin: 5px 0;}
    .stChatMessage.assistant {background-color: #1e1e1e; color: #ffffff; border-radius: 10px; padding: 10px; margin: 5px 0;}
    .answer {background-color: #333333; color: #ffffff; padding: 10px; border-radius: 10px; margin-top: 10px;}
    .sources {background-color: #444444; color: #ffffff; padding: 10px; border-radius: 10px; margin-top: 10px;}
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 AI Chat Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role_class = 'user' if message['role'] == 'user' else 'assistant'
        st.chat_message(role_class).markdown(message['content'])

    prompt = st.chat_input("Type your question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Don't make up an answer.
        Only use the given context.

        Context: {context}
        Question: {question}

        Answer directly. No small talk.
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        with st.spinner("Thinking..."):
            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]
                
                # Extracting page numbers from sources
                source_pages = [f"Page: {doc.metadata.get('page', 'N/A')}" for doc in source_documents]
                sources_text = " | ".join(source_pages)
                
                result_to_show = f"<div class='answer'>{result}</div>"
                result_to_show += f"<div class='sources'><strong>Sources:</strong> {sources_text}</div>"

                st.chat_message('assistant').markdown(result_to_show, unsafe_allow_html=True)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
