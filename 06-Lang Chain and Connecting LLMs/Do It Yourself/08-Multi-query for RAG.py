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
# MAGIC 2.1 Load data from the web (Databricks eBook)

# COMMAND ----------

from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(web_paths=["https://www.databricks.com/discover/pages/optimize-data-workloads-guide",
                                  "https://spark.apache.org/docs/latest/tuning.html"])
blog_doc = loader.load()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Split the document in smaller chunks

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, 
                                               chunk_overlap=0, 
                                               separators=["\n\n\n\n\n\n", "\n\n\n\n", "\n\n"])

doc_splits = text_splitter.split_documents(blog_doc)

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Load an embedding model.\
# MAGIC We want to use BAAI/bge-small-en-v1.5

# COMMAND ----------

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                        cache_folder = "/Volumes/dev/genai_db/models/bge-small-en-v1.5/")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Connect to vector database collection.

# COMMAND ----------

from langchain_chroma import Chroma

db_location = "/tmp/chroma_demo.db"
collection_name = "rag_collection"

vector_db = Chroma(
    embedding_function=embedding_model,
    collection_name=collection_name,
    persist_directory=db_location
)

vector_db.reset_collection()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.5 Load and index your document splits.

# COMMAND ----------

vector_db.add_documents(documents=doc_splits)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Build the Multiquery RAG pipeline
# MAGIC #####3.1 Create a multiquery retriver
# MAGIC 1. Get the user question
# MAGIC 2. Create a document retriever
# MAGIC 3. Create a generation model
# MAGIC 4. Create a multiquery prompt
# MAGIC 5. Create a multquery output parser
# MAGIC 6. Create a multiquery retriever chain
# MAGIC #####3.2 Create a RAG solution
# MAGIC 1. Eliminate duplicate contexts
# MAGIC 2. Join unique contexts
# MAGIC 3. Create a user prompt
# MAGIC 4. Create the final RAG chain
# MAGIC 5. Execute the final RAG chain and get the result

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1.1 Get the user question

# COMMAND ----------

user_question = "How to improve shuffle performance in Spark?"

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1.2 Create a document retriever

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.1.3 Create a generation model\
# MAGIC We want to use gpt-4o-mini from OpenAI

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.1.4 Create a multiquery prompt

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.1.5 Create a multquery output parser

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.1.6 Create a multiquery retriever chain

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1.6.1 Create and test multiquery chain

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.1.6.2 Create and test multiquery retriever chain

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2.1 Eliminate duplicate contexts

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2.2 Join unique contexts

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2.3 Create a user prompt

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2.4 Create final RAG Chain

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2.5 Execute the final RAG chain and get the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>