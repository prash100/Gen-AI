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
comment_2 = "I am tired of calling the customer support. I am not getting any resolution"
comment_3 = "The customer support has done good job to resolve my problem"

# COMMAND ----------

# MAGIC %md
# MAGIC Q1: Perform sentiment analysis on given customer comments.
# MAGIC 1. Do not use pipeline function
# MAGIC 2. Use Auto Classes
# MAGIC 3. Apply distilbert-base-uncased-finetuned-sst-2-english model from huggingface hub

# COMMAND ----------

# MAGIC %md
# MAGIC Step 1 - Create a tokenizer and prepare input tokens

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 1.1 How to create tokens from input string?

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 1.2 How to encode tokens?

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 1.3 How to decode tokens?

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 1.4 How to prepare the inputs using the tokenizer?

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Step 2 - Create an use a model to generate raw outputs

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Step 3 - Convert raw outputs to probabilities

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Step 4 - Format and display the results

# COMMAND ----------


    

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>