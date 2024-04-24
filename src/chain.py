from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from src.prompt_template import *

# from langchain.chains import RetrievalQA





# def create_chain(llm, vector_store, prompt):
def get_conversational_chain():
      model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.1)

      prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

      chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
      print('Chain created')
      return chain



