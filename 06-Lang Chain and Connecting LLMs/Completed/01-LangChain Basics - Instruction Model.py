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
os.environ["OPENAI_API_KEY"]="Your OpenAI API Key>"

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Create and run LLM model

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Run with simple prompt

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1.1 How to use Huggingface LLM with LangChain\
# MAGIC We want to use meta-llama/Llama-3.2-1B-Instruct

# COMMAND ----------

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=pipe)

output_str = llm.invoke("How to optimize Apache Spark performance?")

# COMMAND ----------

output_str

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1.2 How to use OpenAI LLM with Langchain\
# MAGIC We want to use gpt-3.5-turbo-instruct

# COMMAND ----------

from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=50)
output_str = llm.invoke("How to optimize Apache Spark performance?")

# COMMAND ----------

output_str

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Run with prompt template

# COMMAND ----------

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """Answer the given question in less than 50 words.
    Question: {question}"""
)

prompt_value = prompt.invoke({"question": "How to optimize Apache Spark performance?"})

output_str = llm.invoke(prompt_value)

# COMMAND ----------

output_str

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Chain the prompt with LLM

# COMMAND ----------

llm_chain = prompt | llm
output_str = llm_chain.invoke({"question": "How to optimize Apache Spark performance?"})

# COMMAND ----------

output_str

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>

# COMMAND ----------

# MAGIC %md
# MAGIC