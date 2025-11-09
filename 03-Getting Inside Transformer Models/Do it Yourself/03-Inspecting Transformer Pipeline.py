# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ####Using Transformer LLMs for Sentiment Analysis

# COMMAND ----------

comment_1 = "This is an awesome course for beginners"
comment_2 = "I am tired of calling the customer support and not getting any resolution"
comment_3 = "The customer support has done good job to resolve my problem"

# COMMAND ----------

# MAGIC %md
# MAGIC Q2: Perform sentiment analysis on given customer comments.
# MAGIC 1. Use the pipeline function
# MAGIC 2. Apply distilbert-base-uncased-finetuned-sst-2-english model from huggingface hub
# MAGIC 3. Extract Model and Tokenizer from the pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Step 1 - Create the pipeline

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Step 2 - Extract a tokenizer and prepare input tokens

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Step 3 - Extract your model from the pipeline

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Step 4 - Convert raw outputs to probabilities

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Step 5 - Format and display the results

# COMMAND ----------

            

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>