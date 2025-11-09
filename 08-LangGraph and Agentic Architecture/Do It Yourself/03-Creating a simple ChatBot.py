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
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]="<Your Langchain API Key>"

# COMMAND ----------

# MAGIC %md 
# MAGIC 1.2 Define a function to show the graph

# COMMAND ----------

def show_app_graph(app):
    import IPython.display as ipd
    try:
        ipd.display(
            ipd.Image(
                app.get_graph().draw_mermaid_png()
                )
            )
    except Exception:
       app.get_graph().print_ascii() 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Create a Graph
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Create a data structure for state that tracks all messages

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Create an LLM to answer user questions

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Write a graph node function to answer user questions

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Create a the graph with nodes and edges

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Execute the graph with tracing

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Create a convenient function to chat with the LLM graph

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Setup some messages

# COMMAND ----------

instruction = """You are a helpful data analyst who generates SQL queries to answer my questions. Assume you have a table named sales with columns product_id, product_name, and sales_amount. Generate SQL query for my question using this sales table."""

question_1 = "What is the total sales for each product?"
question_2 = "Please explain the SQL query which you just generated."

# COMMAND ----------

# MAGIC %md
# MAGIC 3.3 Give some instruction

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>