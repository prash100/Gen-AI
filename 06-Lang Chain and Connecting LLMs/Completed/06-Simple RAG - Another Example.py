# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ####1. Setup your Environment Variables

# COMMAND ----------

# MAGIC %md
# MAGIC 1.1 Set your API Key

# COMMAND ----------

import os

os.environ["OPENAI_API_KEY"]="<Your OpenAI API Key>"


# COMMAND ----------

# MAGIC %md
# MAGIC 1.2 Get your questions ready

# COMMAND ----------

question1 = "What is AQE?"
question2 = "How to check whether AQE is enabled?"
question3 = "How AQE optimizes Spark SQL?"
question4 = "How Caching optimizes Spark SQL?"

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. RAG Data Prepration
# MAGIC 1. Load the data from the given URL\
# MAGIC https://spark.apache.org/docs/latest/sql-performance-tuning.html
# MAGIC 2. Create splits
# MAGIC 3. Load an embedding model (text-embedding-3-small from OpenAI)
# MAGIC 4. Connect to Vector database
# MAGIC 5. Load you document splits to vector database

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Load data from the given URL\
# MAGIC https://spark.apache.org/docs/latest/sql-performance-tuning.html

# COMMAND ----------

from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(web_path= "https://spark.apache.org/docs/latest/sql-performance-tuning.html",
                       bs_kwargs={"parse_only": bs4.SoupStrainer(id=("content"))})
blog_doc = loader.load()                       

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Split the document in smaller chunks

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                               separators=["\n\n\n\n", "\n\n\n"])

doc_splits = text_splitter.split_documents(blog_doc)                                             

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Load an embedding model.\
# MAGIC We want to use text-embedding-3-small from OpenAI

# COMMAND ----------

# The OpenAIEmbeddings is currently not using the API Key if it is not set in the Environment Variable
# Hence, we will be using another embedding model

'''
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
'''

# COMMAND ----------

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                        cache_folder = "/Volumes/dev/genai_db/models/bge-small-en-v1.5/")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Connect to vector database collection.

# COMMAND ----------

from langchain_chroma import Chroma

vector_db = Chroma(
    embedding_function=embedding_model,
    persist_directory="/tmp/chroma_demo.db",
    collection_name="rag_collection"
)

vector_db.reset_collection()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.5 Load and index your document splits.

# COMMAND ----------

vector_db.add_documents(documents=doc_splits)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Build the RAG pipeline
# MAGIC 1. Create and test a document retriever for your vector database.
# MAGIC 2. Load a generation model.
# MAGIC 3. Create a chat prompt template
# MAGIC 4. Create a function for combining multiple splits
# MAGIC 5. Create a RAG chain
# MAGIC 6. Execute the RAG chain and get the result

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1.1 Create and test a document retriever for your vector database.

# COMMAND ----------

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1.2 Test the retriver

# COMMAND ----------

search_result = retriever.invoke(question1)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Load the generation model.\
# MAGIC We want to use gpt-4o-mini OpenAI model for generation

# COMMAND ----------

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=100)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.3.1 Create an augumented chat prompt template.

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [("system", "Answer to the question using the given context."),
     ("human", "Context: {context}"),
     ("human", "Question: {question}")]
)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.4 Create a function for combining multiple splits

# COMMAND ----------

def join_docs(docs):
    return "\n".join(doc.page_content.strip() for doc in docs)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.5 Create a RAG chain

# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

rag_chain = (
    { "context": itemgetter("question") | retriever | join_docs,
      "question": itemgetter("question")
    } | prompt | llm | StrOutputParser()
)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.6 Execute the RAG chain and get the result

# COMMAND ----------

user_question = question2
ai_message = rag_chain.invoke({"question": user_question})

# COMMAND ----------

ai_message

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>