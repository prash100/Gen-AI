# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ####Using Transformer LLMs for Named Entity Recognition

# COMMAND ----------

# MAGIC %md
# MAGIC Q1: Some english texts are given below. Identify named entities such as persons and places.
# MAGIC 1. Use ner task
# MAGIC 2. Apply bert-large-cased-finetuned-conll03-english model from huggingface

# COMMAND ----------

text_1 = """
My name is Prashant Kumar Pandey and I work at ScholarNest in Bangalore.
"""
text_2 = """
India vs Australia, 2nd Day Warm UP Match Live Cricket Score: Continuous drizzle in Canberra on Saturday robbed India of much needed pink-ball practice on day one of their two-day warm-up fixture at the Manuka Oval against Australia.
"""

# COMMAND ----------

from transformers import pipeline
import pandas as pd

named_entity_recognizer = pipeline("ner", grouped_entities=True)
results = named_entity_recognizer([text_1, text_2])
display(pd.DataFrame(results))

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>