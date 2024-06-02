import os
import pickle
import time
import streamlit as st
import langchain_helper as lh
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv

#Loads environment variables from .env file
load_dotenv()

st.title("Equity Research Tool")
st.sidebar.title("News Article URL's")

urls=[]
filePath="faiss_store_openai.pkl"

mainPlaceHolder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

urlClicked = st.sidebar.button("Process URL's")

if(urlClicked):
    mainPlaceHolder.text("Data Loading Started...")
    # Loading Data
    urlLoader = UnstructuredURLLoader(urls=urls)
    # urlLoader = TextLoader("WebData.txt")
    data = urlLoader.load()
    # Splitting Data
    recChrTxtSplitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n","."," "], chunk_size=1000)
    mainPlaceHolder.text("Text Splitting Started...")
    # time.sleep(2)
    docs= recChrTxtSplitter.split_documents(data)
    # Create Embeddings
    embeddings = OpenAIEmbeddings()
    vectorStoreOpenAI = FAISS.from_documents(docs, embeddings)
    mainPlaceHolder.text("Vector Embedding Started...")
    # Save the FAISS Index to a Pickle File
    with open(filePath, "wb") as f:
        pickle.dump(vectorStoreOpenAI, f)
    print("DONE!!!")

query = mainPlaceHolder.text_input("Question: ")
if query:
    if os.path.exists(filePath):
        with open(filePath,"rb") as f:
            vectorStore= pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.subheader(result["answer"])

            #Display Sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources")
                sourcesList = sources.split("\n")
                for source in sourcesList:
                    st.write(source)



