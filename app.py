from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv

#API 키 불러오기
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

#FAISS 벡터 불러오기
db = FAISS.load_local("faq_vector", OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
retriever = db.as_retriever()

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

st.set_page_config(page_title="서울 FAQ 챗봇")
st.header("📘 서울 FAQ 챗봇")

query = st.text_input("내용을 입력하세요:")

if query:
    try:
        answer = qa_chain.run(query)
        st.write("💬 답변:", answer)
    except Exception as e:
        st.error(f"에러가 발생했습니다 위키를 참조해 주세요: {e}")
