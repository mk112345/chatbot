import os
from langchain_community.document_loaders import UnstructuredURLLoader, MergedDataLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import logging
import io



logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s:'
)


# def load_data_source(loaded_files):
#     print('Inside load_data_source')
#     all_loaders = []
#     for loaded_file in loaded_files:
#         print('loaded_file - ', loaded_file)
#         temp_file = create_temp_file(loaded_file)
       
#         loader = get_loader_by_file_extension(temp_file)
#         # print('loader - ', loader)
#         all_loaders.append(loader)
        
#     loader_all = MergedDataLoader(loaders=all_loaders) 
#     data = loader_all.load()
#     return data
directory_path=r'C:\Users\mayur\Desktop\newset_multi\data'
def load_data_source(directory_path):
    print('Inside load_data_source')
    all_loaders = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        print('file_path - ', file_path)
        with open(file_path, "rb") as file:
            loaded_file = io.BytesIO(file.read())
            temp_file = create_temp_file(loaded_file)
            loader = get_loader_by_file_extension(temp_file)
            all_loaders.append(loader)
        
    loader_all = MergedDataLoader(loaders=all_loaders) 
    data = loader_all.load()
    return data  


def get_data_chunks(data):
    recursive_char_text_splitter=RecursiveCharacterTextSplitter(
                                                chunk_size=500,
                                                chunk_overlap=50)
    documents=recursive_char_text_splitter.split_documents(data)
    # print('documents - ', documents)
    print('documents type - ', type(documents))
    print('documents length - ', len(documents))
    return documents




def create_temp_file(loaded_file):
    temp_file = f"./tmp/{loaded_file.name}"
    with open(temp_file, "wb") as file:
        file.write(loaded_file.getvalue())
    return temp_file

def get_loader_by_file_extension(temp_file):
    file_split = os.path.splitext(temp_file)
    file_extension = file_split[1]
    if file_extension == '.pdf':
        loader = PyPDFLoader(temp_file)
        logging.info('Loader Created for PDF file')
    elif file_extension == '.txt':
        loader = TextLoader(temp_file)
        logging.info('Loader Created for txt file')
    elif file_extension == '.csv':
        loader = CSVLoader(temp_file)
        logging.info('Loader Created for csv file')
    else:
        loader = UnstructuredFileLoader(temp_file)
        logging.info('Loader Created for unstructured file')
    return loader