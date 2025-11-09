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
# MAGIC #### 2. Preparing Data for Vector database

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Load data from text file

# COMMAND ----------

from langchain_community.document_loaders import TextLoader
loader = TextLoader("/Volumes/dev/genai_db/raw_data/test_data/markdown_sample.md")
text_doc = loader.load()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Split your data into smaller chunks

# COMMAND ----------

from langchain_text_splitters import MarkdownHeaderTextSplitter
headers_to_split_on = [
    ("#", "Heading_1"),
    ("##", "Heading_2")
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
doc_splits = markdown_splitter.split_text(text_doc[0].page_content)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Prepare and load data into vector database

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Prepare your embedding model\
# MAGIC Use text-embedding-3-small from OpenAI or BAAI/bge-small-en-v1.5 from Huggingface

# COMMAND ----------

# The OpenAIEmbeddings is currently not using the API Key if it is not set in the Environment Variable
# Hence, we will be using another embedding model

'''
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
'''

# COMMAND ----------

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                        cache_folder = "/Volumes/dev/genai_db/models/bge-small-en-v1.5/")

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Prepare your vector database connection

# COMMAND ----------

from langchain_milvus import Milvus

db_location = "/tmp/milvus_demo.db"
collection_name = "rag_collection"

vector_db = Milvus(
    embedding_function=embedding_model,
    connection_args={"uri": db_location},
    collection_name=collection_name,
    drop_old=True,
    auto_id=True,
    consistency_level="Strong",
    index_params={"metric_type": "IP"}
)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.3 Add documents to vector database

# COMMAND ----------

vector_db.add_documents(documents=doc_splits)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.4 Get your runnable document retriever

# COMMAND ----------

retriver = vector_db.as_retriever(search_kwargs={"k":2})

# COMMAND ----------

# MAGIC %md
# MAGIC 3.5 Start using retriever

# COMMAND ----------

search_result = retriver.invoke("What is AQE?")

# COMMAND ----------

search_result

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>