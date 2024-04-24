from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
# from langchain.vectorstores import Pinecone
from flask import Flask, request, jsonify, render_template
from langchain.chains import RetrievalQA

from dotenv import load_dotenv



load_dotenv()
import os

GOOGLE_API_KEY = os.getenv('Google_API_Key')

os.environ['Google_API_Key'] = GOOGLE_API_KEY



app = Flask(__name__)

# load_dotenv()
def load_pdf(directory):
    pdf_files = glob.glob(os.path.join(directory, '*.pdf'))
    return pdf_files

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    directory = r"C:\Users\mayur\Desktop\new_data\data"
    pdf_files = load_pdf(directory)
    
    # Get text from PDFs
    text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(text) 

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    get_vector_store(text_chunks)
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    
    return {"answer": response["output_text"]}
    



@app.route('/')
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = user_input(input)
    print("Response : ", result["answer"])
    return str(result["answer"])
    
  

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
