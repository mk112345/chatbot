from langchain.embeddings import GoogleGenerativeAIEmbeddings


def create_embeddings():
    embeddings=GoogleGenerativeAIEmbeddings(
            model_name='models/embedding-001', 
            # model_kwargs={'device':'cpu'}
    )
    return embeddings