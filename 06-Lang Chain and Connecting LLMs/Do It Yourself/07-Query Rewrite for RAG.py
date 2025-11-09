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
# MAGIC 5. Load your document splits to vector database

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Load data from the source file 

# COMMAND ----------

from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(web_path="https://spark.apache.org/docs/latest/sql-performance-tuning.html",
                       bs_kwargs={"parse_only": bs4.SoupStrainer(id=("content"))})
blog_doc = loader.load()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Split the document in smaller chunks

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0, separators=["\n\n\n\n", "\n\n\n"])

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
# MAGIC ####3. Build a simple RAG pipeline
# MAGIC 1. Get the original user question
# MAGIC 2. Create and test a document retriever for your vector database.
# MAGIC 3. Load a generation model.
# MAGIC 4. Create a prompt template
# MAGIC 5. Create a function for combining multiple splits
# MAGIC 6. Create a RAG chain
# MAGIC 7. Execute the RAG chain and get the result

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Get the original user question

# COMMAND ----------

user_question = "I am Prashant and I am an expert in Programming. I am enjoying my coffee at this time in the morning and simultaniously playing with my computer and learn How Caching optimizes Spark SQL?"

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2.1 Create a document retriever for your vector database.

# COMMAND ----------

retriever = vector_db.as_retriever(search_kwargs={"k": 1})

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2.2 Test the retriver

# COMMAND ----------

search_result = retriever.invoke(user_question)
print(search_result[0].page_content)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.3 Load a generation model.\
# MAGIC We want to use gpt-4o-mini OpenAI model for generation.

# COMMAND ----------

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=100)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.4 Create an augumented prompt template.

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [("system", "Answer to the question using the given context."),
     ("human", "Context: {context}"),
     ("human", "Question: {question}")]
)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.5 Create a function for combining multiple splits

# COMMAND ----------

def join_docs(docs):
    return "\n".join(doc.page_content.strip() for doc in docs)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.6 Create a RAG chain

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

result = rag_chain.invoke({"question": user_question})
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ####4. Build a Query rewrite RAG pipeline
# MAGIC 1. Get the original user question - Done
# MAGIC 2. Create and test a document retriever for your vector database. - Done
# MAGIC 3. Load a generation model. - Done
# MAGIC 4. Create a prompt template - Done
# MAGIC 5. Create a function for combining multiple splits - Done
# MAGIC 6. Create a query rewrite prompt
# MAGIC 7. Create a query rewrite chain
# MAGIC 8. Create final RAG chain
# MAGIC 9. Execute the final RAG chain and get the result

# COMMAND ----------

# MAGIC %md
# MAGIC 4.6 Create a query rewrite prompt

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 4.7.1 Create a query rewrite chain

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 4.7.2 Test the query rewrite chain

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 4.7.3 Test the retriver with rewrite question

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 4.8 Create final RAG chain

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 4.9 Execute the final RAG chain and get the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>