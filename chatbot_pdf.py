import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
import openai
import streamlit as st
from streamlit_chat import message

folderPath = './docData'

if os.path.exists(folderPath):

    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=folderPath, embedding_function=embedding)


else:

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)

    print('분할된 텍스트 개수: ',len(texts))
    
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=folderPath)


retriever = vectordb.as_retriever(search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo",temperature=0),
                                       chain_type="stuff", 
                                       retriever = retriever, 
                                       return_source_documents = True)

st.title("PDF 문서 학습 테스트")

user_id = st.text_input(label="User Name", value="")


if user_id == "TEST_KEY":
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    placeholder = st.empty()

    with st.form('form', clear_on_submit=True):
        user_input = st.text_input("Question: ",'')
        submitted = st.form_submit_button('Send')

    # text_source = ""

    if submitted and user_input:
        chatbot_ans = qa_chain(user_input)

        # for idx, source in enumerate(chatbot_ans["source_documents"]):
        #     if idx == 0: 
        #         text_source += "\n\n 문서출처: " + source.metadata['source'] + " 페이지: "+ str(source.metadata['page']) + " "
        #     else:
        #         text_source += "\n 문서출처: " + source.metadata['source'] + " 페이지: "+ str(source.metadata['page']) + " "

        st.session_state['past'].append(user_input)
        # st.session_state['generated'].append(chatbot_ans['result'].strip() + text_source)
        st.session_state['generated'].append(chatbot_ans['result'].strip())

    with placeholder.container():
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state['generated'][i], key=str(i))



