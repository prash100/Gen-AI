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
# MAGIC 2.1 Create following data structures
# MAGIC 1. A data model for an structured LLM to produce SQL expression
# MAGIC 2. A data structure for passing state across graph nodes

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Create following prompts
# MAGIC 1. A prompt for generating SQL expression from user question
# MAGIC 2. A prompt for explaining generated sql expression

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Create following LLM models
# MAGIC 1. A model to generate SQL expression
# MAGIC 2. A model to generate SQL explanation

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Write a graph node function for SQL generation

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.5 Write a graph node function to generate SQL explanation

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.6 Create a the graph with nodes and edges

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.7 Execute the graph with tracing

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.8 Print the result

# COMMAND ----------

# MAGIC %md
# MAGIC 2.8.1 Print the sql query

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2.8.2 Print the sql explanation

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>