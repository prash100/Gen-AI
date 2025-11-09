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
# MAGIC 2.1 Load data from text file\
# MAGIC /Volumes/dev/genai_db/raw_data/test_data/markdown_sample.md

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Split your data into smaller chunks

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Prepare and load data into vector database

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Prepare your embedding model\
# MAGIC Use BAAI/bge-small-en-v1.5 from Huggingface

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Prepare your vector database connection

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.3 Add documents to vector database

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.4 Get your runnable document retriever

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.5 Start using retriever

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>