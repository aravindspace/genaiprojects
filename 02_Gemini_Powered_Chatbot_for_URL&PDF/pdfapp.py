import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader


# ### Langsmith tracking

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Function to load data
def get_pdf_text(pdf):
    pdf_reader=PdfReader(pdf)
    text=""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function for to chunking data(Transform)
def get_text_chunks(text):
    splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    return splitter.split_text(text)

# Function for coversational chain
def get_conversational_chain():
    prompt_template= """ Answer the question as deatailed as possible from the provided context. Make sure to provide all the details.
    If the answer is not in the provided context, just say, " Answer is not available",Don't provide worng answer

    context:\n{context}\n
    Question:\n{question}\n
    Answer:

    """

    model=ChatGoogleGenerativeAI(model="models/gemini-2.5-flash",temperature=0.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    return load_qa_chain(llm=model,prompt=prompt,chain_type="stuff")


# --- Streamlit UI Setup ---
st.set_page_config(page_title="🧠 InteractiveQA - URL Reader", layout="wide")
st.markdown("<h1 style='text-align: center; color: #3A7CA5;'>🌐 Interactive Q&A App with Google Gemini</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask questions based on content from a public URL. Powered by LangChain + Gemini + FAISS.</p>", unsafe_allow_html=True)
st.markdown("---")



# --- Sidebar ---
with st.sidebar:
    st.header("🔐 Google API Key")
    api_key = st.text_input("Enter your Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"]=api_key

    st.markdown("---")
    st.header("🔗 Upload a PDF")
    uploaded_file=st.sidebar.file_uploader("Choose a pdf file",type="pdf")

    st.markdown("---")
    st.info("Your key is not stored. It is only used during this session.")

# --- Main Input ---
st.subheader("💬 Ask a Question")
user_question = st.text_input("Enter your question below and click 'Get Answer'")


# Main code
if st.button("🚀 Get Answer") and api_key and uploaded_file:
    try:
        with st.spinner("🔍 Fetching answer from webpage..."):
            text=get_pdf_text(uploaded_file)
            chunks=get_text_chunks(text)
            embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store=FAISS.from_texts(chunks,embedding=embeddings)
            # if want to store it in vectordb in local we can
            # vector_store.save_local("Temp")
            docs=vector_store.similarity_search(user_question)
            chain=get_conversational_chain()
            response=chain({"input_documents":docs,"question":user_question})

            
            st.success("✅ Answer Generated!")
            st.markdown("### ❓ Your Question")
            st.write(user_question)
            st.header("Generated Answer:")
            # st.write("DEBUG RESPONSE:", response)
            # Display the response line by line
            for line in response["output_text"].split("\n"):
                if line.strip():
                    st.write(line)
    except Exception as e:
        st.error(f"🚨 An error occurred: {e}")
else:
    st.warning("⚠️ Please enter both the API key and a valid PDF to proceed.")
