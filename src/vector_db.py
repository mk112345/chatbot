from langchain.vectorstores import FAISS
import pandas as pd
from langchain.chains.question_answering import load_qa_chain
from src.prompt_template import *
from src.chain import *
from constants import *
# from main import st

# VECTOR_DB_DIRECTORY=r'C:\Users\mayur\Desktop\ofc_new_data\newset_multi\faiss_index'
def store_data_in_vectordb(documents, embeddings):
    existing_knowledge_base = None
    try:
        existing_knowledge_base = load_vectordb(VECTOR_DB_DIRECTORY, embeddings)
        print('current_vectordb - ', convert_vectordb_to_df(existing_knowledge_base))
    except :
        print('Error in loading the existing vector DB')

    new_knowledge_base =FAISS.from_documents(documents, embeddings)
    print('new_knowledge_base - ', convert_vectordb_to_df(new_knowledge_base))

    if existing_knowledge_base:
        print('Merging the new data base into the existing knowledge base')
        existing_knowledge_base.merge_from(new_knowledge_base)
        existing_knowledge_base.save_local(VECTOR_DB_DIRECTORY)
        print('updated_knowledge_base - ', convert_vectordb_to_df(existing_knowledge_base))
    else:
        print('Saving the new data base')
        new_knowledge_base.save_local(VECTOR_DB_DIRECTORY)
        print('updated_knowledge_base - ', convert_vectordb_to_df(new_knowledge_base))

    final_loaded_knowledge_base = load_vectordb(VECTOR_DB_DIRECTORY, embeddings)
    print('final_loaded_knowledge_base - ', convert_vectordb_to_df(final_loaded_knowledge_base))
   

    return final_loaded_knowledge_base


def load_vectordb(stored_directory, embeddings):
    loaded_vector_db = FAISS.load_local(stored_directory, embeddings)
    return loaded_vector_db


## TODO : Need to visit code to add documents to existing vectorDB
def get_vector_store(text_chunks, embeddings):
    #Convert the Text Chunks into Embeddings and Create a FAISS Vector Store
    knowledge_base =FAISS.from_documents(text_chunks, embeddings)
    knowledge_base.add_documents(embeddings)
    print('INSIDE get_vector_store :: knowledge_base created')
    return knowledge_base 


import pandas as pd
def convert_vectordb_to_df(vectorDb):
    try:
        vector_dict = vectorDb.docstore._dict
        data_rows = []

        for k in vector_dict.keys():
            doc_name = vector_dict[k].metadata['source'].split('/')[-1]
            if 'page' not in vector_dict[k].metadata:
                page_number = 1
            else:
                page_number = vector_dict[k].metadata['page'] + 1
            content =  vector_dict[k].page_content
            data_rows.append({"chunk_id": k, "document": doc_name, "page": page_number, "content":content})

        vector_df = pd.DataFrame(data_rows)
        print(vector_df)
        return vector_df
    except Exception as e :
        print('Error in convert_vectordb_to_df - ', e)
        return None


def save_vectordb_to_local(vectordb, vector_db_directory):
    vectordb.save_local(vector_db_directory)

def delete_document(vectordb, document):
    vector_df = convert_vectordb_to_df(vectordb)
    chunk_list = vector_df.loc[vector_df['document'] == document]['chunk_id'].tolist()
    vectordb.delete(chunk_list)

def refresh_model(vectordb, llm):
    retriever = vectordb.as_retriever()
    # model = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retruever = retriever)
    model=load_qa_chain(model, chain_type="stuff",retriever=retriever)
    return model


def get_similiar_docs(vector_db, query,k=1,score=False):
  
    vectordb_results = list()
    if score:
        similar_docs_with_score = vector_db.similarity_search_with_score(query,k=k)
        for doc, score in similar_docs_with_score:
            print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
            vectordb_results.append(doc.page_content)
    else:
        similar_docs = vector_db.similarity_search(query,k=k)
        for doc in similar_docs:
            vectordb_results.append(doc.page_content)

    return vectordb_results