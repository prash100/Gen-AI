# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ####Using Transformer LLMs for Language Translation

# COMMAND ----------

# MAGIC %md
# MAGIC Q1: Some english texts are given below. Translate these texts to Spanish.
# MAGIC 1. Use translation task
# MAGIC 2. Apply Helsinki-NLP/opus-mt-en-es model from huggingface

# COMMAND ----------

news_1 = "US universities requests Indian and foreign students to rejoin collage"
news_2 = "Hamas representatives will go to Cairo on Saturday for talks on a possible ceasefire in Gaza"
news_3 = "Staying updated with global news is essential in today’s rapidly changing world"
news_4 = "President Trump hosted a lavish dinner at his club and also invited Elon Musk"

# COMMAND ----------

from transformers import pipeline
import pandas as pd
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
results = translator([news_1, news_2, news_3, news_4])
display(pd.DataFrame(results))

# COMMAND ----------

# MAGIC %md
# MAGIC Q2: Translate given texts to Hindi.
# MAGIC 1. Use translation task
# MAGIC 2. Apply Helsinki-NLP/opus-mt-en-hi model from huggingface

# COMMAND ----------

from transformers import pipeline
import pandas as pd
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
results = translator([news_1, news_2, news_3, news_4])
display(pd.DataFrame(results))

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>