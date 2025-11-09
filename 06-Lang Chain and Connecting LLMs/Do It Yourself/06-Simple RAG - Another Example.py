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

                     

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Split the document in smaller chunks

# COMMAND ----------

                                           

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Load an embedding model.\
# MAGIC We want to use text-embedding-3-small from OpenAI

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Connect to vector database collection.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.5 Load and index your document splits.

# COMMAND ----------



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



# COMMAND ----------

# MAGIC %md
# MAGIC 3.1.2 Test the retriver

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Load the generation model.\
# MAGIC We want to use gpt-4o-mini OpenAI model for generation

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.3.1 Create an augumented chat prompt template.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.4 Create a function for combining multiple splits

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.5 Create a RAG chain

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.6 Execute the RAG chain and get the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>