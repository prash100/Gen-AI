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
# MAGIC ####2. Create a pipeline using meta-llama/Llama-3.2-1B-Instruct

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 3 Find answers to the given questions

# COMMAND ----------

question1 = "What is AQE?"
question2 = "How to check whether AQE is enabled?"
question3 = "How AQE optimizes Spark SQL?"   

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Create a prompt and find answer to question1.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Create a prompt and find answer to question2.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.3 Create a prompt and find answer to question3.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Find answers to the given questions using the given context.

# COMMAND ----------

context = """Adaptive Query Execution (AQE) is an optimization technique in Spark SQL that makes use of the runtime statistics to choose the most efficient query execution plan, which is enabled by default since Apache Spark 3.2.0. Spark SQL can turn on and off AQE by spark.sql.adaptive.enabled as an umbrella configuration. As of Spark 3.0, there are three major features in AQE: including coalescing post-shuffle partitions, converting sort-merge join to broadcast join, and skew join optimization."""

# COMMAND ----------

# MAGIC %md
# MAGIC 4.1 Create a prompt and find answer to question1.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 4.2 Create a prompt and find answer to question2.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 4.3 Create a prompt and find answer to question3.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>