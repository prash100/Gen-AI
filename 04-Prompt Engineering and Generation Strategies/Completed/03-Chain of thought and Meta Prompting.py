# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ####Using Transformer LLMs for Chain of Throught prompting

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Create a pipeline for text-generation using a model Qwen/Qwen2.5-1.5B-Instruct

# COMMAND ----------

from transformers import pipeline, set_seed
pipe = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
set_seed(45)

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Create a prompt for a problem solving which could automatically trigger chain of thought solution.

# COMMAND ----------

prompt="""
I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. How many apples did I remain with?
"""
result = pipe(prompt, min_new_tokens=5, max_new_tokens=200)
print(result[0]["generated_text"])

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Structure the prompt to improve the result and ensure CoT prompting.

# COMMAND ----------

prompt="""
Give answer to the question using the problem statement. Let's think this step-by-step.
Problem Statement: I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. 
Question: How many apples did I remain with?
Answer:
"""
result = pipe(prompt, min_new_tokens=5, max_new_tokens=200)
print(result[0]["generated_text"])

# COMMAND ----------

# MAGIC %md
# MAGIC ####Use meta prompting along with CoT with a solution structure.

# COMMAND ----------

prompt="""
Give answer to the question using the problem statement. Let's think this step-by-step. 
Provide the answer using the given solution structure.
Solution Structure:
1. Begin with the response "Let's think this step-by-step"
2. Follow the reasoning steps, ensuring the solution is broken down clearly and logically.
3. End the solution with Final Answer
4. Finally state "The answer is [Final Answer to the problem]"
Problem Statement: I went to the market and bought 10 apples. I gave 2 apples to the neighbor and 2 to the repairman. I then went and bought 5 more apples and ate 1. 
Question: How many apples did I remain with?
Answer:
"""
result = pipe(prompt, min_new_tokens=5, max_new_tokens=200)
print(result[0]["generated_text"])

# COMMAND ----------

# MAGIC %md
# MAGIC Learn more about prompt engineering and different prompting techniques at https://www.promptingguide.ai

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>