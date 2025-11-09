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

os.environ["HF_TOKEN"]="<Your Huggingface Token>"

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Load the data and prepare for embedding

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Load data from the source file 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Split the document in smaller chunks

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Prepare and test the embedding model.\
# MAGIC We want to use the BAAI/bge-small-en-v1.5 model for embeddings.\
# MAGIC Create a function to return embeddings of a given chunk. We want to use this function later.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Find the number of embedding dimensions for later usage.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 3 Prepare the Vector Database

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Create Vector Database. We want to use pymilvus vector database.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Create a collection for your embeddings

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Prepare and load your embedding collection

# COMMAND ----------

# MAGIC %md
# MAGIC 4.1 Prepare an embedding table

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 4.2 Load embedding table into collection

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####5. Build the RAG pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC 5.1 Load the generation model pipeline. We want to use meta-llama/Llama-3.2-1B-Instruct model for generation

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 5.2 Get your question.

# COMMAND ----------

question1 = "What is AQE?"
question2 = "How to check whether AQE is enabled?"
question3 = "How AQE optimizes Spark SQL?"
question4 = "How Caching optimizes Spark SQL?"

# COMMAND ----------

# MAGIC %md
# MAGIC 5.3 Search related contexts for the question in the vector database

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 5.4 Prepare the context

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 5.5 Create an augumented prompt to answer the given question using the retrived context.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 5.6 Generate the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>