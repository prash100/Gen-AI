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
# MAGIC 1.1 Setup your API Key

# COMMAND ----------

import os

os.environ["OPENAI_API_KEY"]="<Your OpenAI API Key>"

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
# MAGIC #### 2. RAG Data Prepration
# MAGIC 1. Load the data
# MAGIC 2. Create splits
# MAGIC 3. Load an embedding model
# MAGIC 4. Connecto to Vector database
# MAGIC 5. Load you document splits to vector database

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Load data from the PDF

# COMMAND ----------

from langchain_community.document_loaders import PyPDFLoader

python_loader = PyPDFLoader("/Volumes/dev/genai_db/raw_data/test_data/PromptTemplates_Python.pdf")
python_doc = python_loader.load()

js_loader = PyPDFLoader("/Volumes/dev/genai_db/raw_data/test_data/PromptTemplates_JS.pdf")
js_doc = js_loader.load()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Create Splits

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=300, separators=["\n\n\n", "\n\n"])

splitted_python_docs = splitter.split_documents(python_doc)
splitted_js_docs = splitter.split_documents(js_doc)

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Load an embedding model.\
# MAGIC We want to use text-embedding-3-small from OpenAI

# COMMAND ----------

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                        cache_folder = "/Volumes/dev/genai_db/models/bge-small-en-v1.5/")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Connect to vector database collection.

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4.1 Create DB for Python docs and load data

# COMMAND ----------

from langchain_chroma import Chroma

db_location = "/tmp/python_docs.db"
collection_name = "rag_collection"

python_docs_db = Chroma(
    embedding_function=embedding_model,
    collection_name=collection_name,
    persist_directory=db_location
)

python_docs_db.reset_collection()
python_docs_db.add_documents(documents=splitted_python_docs)

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4.2 Create DB for JS Docs and load data

# COMMAND ----------

from langchain_chroma import Chroma

db_location = "/tmp/js_docs.db"
collection_name = "rag_collection"

js_docs_db = Chroma(
    embedding_function=embedding_model,
    collection_name=collection_name,
    persist_directory=db_location
)

js_docs_db.reset_collection()
js_docs_db.add_documents(documents=splitted_js_docs)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Build the RAG pipeline
# MAGIC 1. Create an llm to choose the route
# MAGIC 2. Create a prompt to choose the route
# MAGIC 3. Create a function to parse the route
# MAGIC 4. Create a function to return the appropriate retriever
# MAGIC 5. Create and test the retriever chain
# MAGIC 6. Create a function to join the retrieved documents
# MAGIC 7. Create a prompt for user question
# MAGIC 8. Create the final rag chain
# MAGIC 9. Run and test the final rag chain

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Create an llm to produce one of the following outputs.
# MAGIC - python_docs
# MAGIC - js_docs
# MAGIC
# MAGIC These outputs will guide us to an appropriate retriever.\
# MAGIC We want to use gpt-4o-mini llm from OpenAI

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Create a prompt to choose the route depending upon the programming language used in the user question

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.3 Create a function to parse the route

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.4 Create a function to return the appropriate retriever

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.5 Create and test the retriever chain

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.6 Create a function to join the retrieved documents 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.7 Create a prompt for user question

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.8 Create the final rag chain

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.9 Run and test the final rag chain

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>