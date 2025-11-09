# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ####Using Transformer LLMs for Text Classification

# COMMAND ----------

# MAGIC %md
# MAGIC Q1: Some english texts are given below. Translate these texts to Spanish.
# MAGIC 1. Use translation task
# MAGIC 2. Apply Helsinki-NLP/opus-mt-en-es model from huggingface
# MAGIC 3. Use the following lables:\
# MAGIC politics, finance, sports, science and technology, pop culture, breaking news.

# COMMAND ----------

article_1 = """
Maharashtra's political landscape shifts Shiv Sena's Eknath Shinde signaling withdrawal from the Chief Minister race, speculation grows around Devendra Fadnavis as a frontrunner. The coalition's dynamics complicate the selection process ahead of the speculated December 5 oath-taking ceremony.
"""
article_2 = """
India vs Australia PM XI, 2nd Day Warm UP Match Live Cricket Score: Continuous drizzle in Canberra on Saturday robbed India of much needed pink-ball practice on day one of their two-day warm-up fixture at the Manuka Oval against Australia Prime Minister's XI, ahead of the day-night second Test in Adelaide.
"""

# COMMAND ----------

from transformers import pipeline
import pandas as pd
zero_shot_classifier = pipeline(
    task="zero-shot-classification",
    model="cross-encoder/nli-deberta-v3-small"
)

results = zero_shot_classifier(
        [article_1, article_2],
        candidate_labels=[
            "politics",
            "finance",
            "sports",
            "science and technology",
            "pop culture",
            "breaking news",
        ],
    )

display(pd.DataFrame(results))

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>