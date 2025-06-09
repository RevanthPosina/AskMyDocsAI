import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import torch
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlchatTemplate import bot_template, user_template, css

try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", timeout=60)
except Exception as e:
    print(f"Error loading tokenizer: {e}")

# Extract raw text from multiple PDFs
def extract_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Split text into overlapping chunks
def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Load Qwen tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B").to("cpu")

# Embedding function class wrapper
class QwenEmbedding(Embeddings):
    def embed_documents(self, texts):
        return [get_embedding(text) for text in texts]

    def embed_query(self, text):
        return get_embedding(text)

# Embedding function using mean pooling
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # Move input tensors to CPU
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding

# Create FAISS vectorstore
def get_vectorstore(text_chunks):
    embedding_fn = QwenEmbedding()
    vectorstore = FAISS.from_texts(text_chunks, embedding_fn)
    return vectorstore

# Creating Conversation Chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    # Do NOT render messages here!

# Streamlit UI
def main():
    load_dotenv()
    st.set_page_config(page_title="MultiPDFChatLM", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDF files :books:")

    # Multi-line input and send button
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_area("Ask a question about your PDFs:", key="user_input", height=50)
        send_clicked = st.form_submit_button("Send")

    if send_clicked and user_question.strip():
        handle_userinput(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Welcome message if no chat history
    if st.session_state.get("chat_history") is None:
        st.write(bot_template.replace("{{MSG}}", "👋 Moshi Moshi, I'm Hinabot, how can I help you?"), unsafe_allow_html=True)

    # Render chat history, most recent first
    if st.session_state.get("chat_history"):
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            idx = len(st.session_state.chat_history) - 1 - i
            if idx % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    with st.sidebar:
        # st.title("Upload PDF files")
        pdf_files = st.file_uploader("Upload your PDF files here", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Process"):
            if pdf_files:
                with st.spinner("Processing..."):
                    raw_text = extract_pdf_text(pdf_files)
                    text_chunks = split_text_into_chunks(raw_text)
                    # st.success("Text split into chunks!")
                    # st.write(text_chunks[:3])  # Show a preview
                    vectorstore = get_vectorstore(text_chunks)
                    st.success("Vectorstore created!")

                    # Creating Conversation Chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)


            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
