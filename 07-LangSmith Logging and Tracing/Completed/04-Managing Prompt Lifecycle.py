# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ####1. Setup your Environment Variables

# COMMAND ----------

import os

os.environ["OPENAI_API_KEY"]="<Your OpenAI API Key>"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]="<Your Langchain API Key>"

# COMMAND ----------

# MAGIC %md
# MAGIC 1.2 Get your questions ready

# COMMAND ----------

question_py = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

question_js = """Why doesn't the following code work:

import { ChatPromptTemplate } from "@langchain/core/prompts";

const chatPrompt = ChatPromptTemplate.fromMessages([
  ["human", "speak in {language}"],
]);

const formattedChatPrompt = await chatPrompt.invoke({
  input_language: "french"
});
"""

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Implement RAG Data Prepration
# MAGIC Create a class for the following functionality
# MAGIC 1. Load data from PDF source
# MAGIC 2. Split documents into smaller chunks
# MAGIC 3. Connect to vector DB and store document chunks
# MAGIC 4. Create a vector DB retriever

# COMMAND ----------

class RagRetriever():
    from langsmith import trace
    def __init__(self):
        self.embedding_model_name = "BAAI/bge-small-en-v1.5"
        self.collection_name = "rag_collection"

    def load_pdf(self, location):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(location)
        return loader.load()

    def split_docs(self, docs):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        with trace(name="split_docs", inputs={"doc_count":len(docs), "docs":docs}) as rt:
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, separators=["\n\n\n", "\n\n"])
            splitted_docs = splitter.split_documents(docs)
            rt.end(outputs={"split_count": len(splitted_docs)})
            return splitted_docs

    def store_docs(self, db_location, docs):
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embedding_model = HuggingFaceEmbeddings(model=self.embedding_model_name,
                                                cache_folder = "/Volumes/dev/genai_db/models/bge-small-en-v1.5/")
        vector_db =  Chroma(
            embedding_function=embedding_model,
            collection_name=self.collection_name,
            persist_directory=db_location)
        vector_db.reset_collection()
        vector_db.add_documents(documents=docs)
        return vector_db        

    def get(self, pdf_location, db_location):
        docs = self.load_pdf(pdf_location)
        splitted_doc = self.split_docs(docs)
        vector_db = self.store_docs(db_location, splitted_doc)
        return vector_db.as_retriever(search_kwargs={"k": 2})


# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Implement RAG Functionality
# MAGIC 1. Create a data model for structured llm output
# MAGIC 2. Create a Router class for the following functionality
# MAGIC     1. Create an llm to identify the route
# MAGIC     2. Create a prompt with routing instructions
# MAGIC     3. Invoke the llm to determine the route
# MAGIC 3. Create a RagBot class for the following functionality
# MAGIC     1. Retrieve the context documents
# MAGIC     2. Combine contextual documents
# MAGIC     3. Create an appropriate q/a prompt
# MAGIC     4. Create an LLM to answer the question
# MAGIC     5. Create an LLM chain
# MAGIC     6. Answer the question

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Create a data model for structured llm output

# COMMAND ----------

from pydantic import BaseModel
from pydantic import Field
from typing import Literal

class Route(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasourse: Literal["python_docs", "js_docs"] = Field(
        description="Given a user question choose which datasource would be most relevant for answering their question"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Create a class for the following functionality
# MAGIC 1. Create an llm to identify the route
# MAGIC 2. Create a prompt with routing instructions
# MAGIC 3. Invoke the llm to determine the route

# COMMAND ----------

class RagRouter():
    from langsmith import traceable    
    def __init__(self):
        self.model_name = "gpt-4o-mini"

    def get_router(self):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=self.model_name, temperature=0, max_tokens=300)
        return llm.with_structured_output(Route)
    
    def get_router_prompt(self):
        from langsmith import Client
        lsm_client = Client()
        return lsm_client.pull_prompt("router_prompt")

    @traceable
    def get_route(self, question):
        router = self.get_router()
        router_prompt = self.get_router_prompt()
        router_chain = router_prompt | router
        route =  router_chain.invoke(question)
        return route.datasourse.lower()

# COMMAND ----------

from langsmith import Client
lsm_client = Client()
prompt = lsm_client.pull_prompt("router_prompt")

# COMMAND ----------

# MAGIC %md
# MAGIC 3.3 Create a RagBot class for the following functionality
# MAGIC 1. Retrieve the context documents
# MAGIC 2. Combine contextual documents
# MAGIC 3. Create an appropriate q/a prompt
# MAGIC 4. Create an LLM to answer the question
# MAGIC 5. Create an LLM chain
# MAGIC 6. Answer the question

# COMMAND ----------

class RagBot():
    def __init__(self):
        self.model_name = "gpt-4o-mini"
        self.py_doc_source = "/Volumes/dev/genai_db/raw_data/test_data/PromptTemplates_Python.pdf"
        self.py_db_location = "/tmp/python_docs.db"
        self.js_doc_source = "/Volumes/dev/genai_db/raw_data/test_data/PromptTemplates_JS.pdf"
        self.js_location = "/tmp/js_docs.db"


    def get_contexts(self, question):
        retriever = RagRetriever()

        router = RagRouter()
        route = router.get_route(question)
        
        if "python_docs" in route:
            py_retriever = retriever.get(pdf_location=self.py_doc_source, 
                                         db_location=self.py_db_location)
            return py_retriever.invoke(question)
        else:
            js_retriever = retriever.get(pdf_location=self.js_doc_source, 
                             db_location=self.js_location)
            return js_retriever.invoke(question)

    def join_docs(self, docs):
        return "\n".join(doc.page_content.strip() for doc in docs)
    
    def get_prompt(self):
        from langsmith import Client
        lsm_client = Client()
        return lsm_client.pull_prompt("rag_prompt")
  
    def get_llm(self):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=self.model_name, temperature=0, max_tokens=300)

    def get_chain(self):
        from operator import itemgetter
        from langchain_core.runnables import RunnableLambda
        prompt = self.get_prompt()
        llm = self.get_llm()

        return (
            {"context": itemgetter("question") | RunnableLambda(self.get_contexts) | RunnableLambda(self.join_docs),
             "question": itemgetter("question")
             } | prompt | llm
            )
        
    def answer(self, question):
        rag_chain = self.get_chain()
        return rag_chain.invoke({"question": question})


# COMMAND ----------

# MAGIC %md
# MAGIC ####4. Execute the RAG Bot to answer questions
# MAGIC 1. Ask a Java Script problem
# MAGIC 2. Ask a Python problem

# COMMAND ----------

# MAGIC %md
# MAGIC 4.1 Ask a Java Script problem

# COMMAND ----------

from langsmith import trace

with trace(project_name="Rag Bot", 
           inputs={"question":question_js },
           name="Rag for js question") as rt:
    bot = RagBot()
    answer = bot.answer(question_js)
    rt.end(outputs={"answer":answer})

# COMMAND ----------

# MAGIC %md
# MAGIC 4.2 Ask a Python problem

# COMMAND ----------

with trace(project_name="Rag Bot", 
           inputs={"question":question_py },
           name="Rag for py question") as rt:
    bot = RagBot()
    answer = bot.answer(question_py)
    rt.end(outputs={"answer":answer})

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>