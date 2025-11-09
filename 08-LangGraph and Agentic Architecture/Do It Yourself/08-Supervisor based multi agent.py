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
# MAGIC #### 2. Create Coder Agent
# MAGIC

# COMMAND ----------


#Define state


#Define Python repl tool


#Create llm


#Define coder node function

#Create coder agent app


# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Create Researcher Agent

# COMMAND ----------


#Define state

#Define search tool

#Create llm

#Define research node function

#Create research agent app


# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Prepare for Supervisor
# MAGIC Note: Supervisor implements routing amongst available agents (researcher, coder).

# COMMAND ----------



#Define structured output data model


#Create structured output model


#Create supervisor prompt for routing


# COMMAND ----------

# MAGIC %md
# MAGIC ####4. Create supervisor architecture

# COMMAND ----------

# MAGIC %md
# MAGIC 4.1 Create required state and functions

# COMMAND ----------


#Define supervisor state


#Define supervisor node function


#Define routing function for conditional edge


# COMMAND ----------

# MAGIC %md
# MAGIC 4.2 Create a the graph with nodes and edges

# COMMAND ----------




# COMMAND ----------

# MAGIC %md
# MAGIC ####5. Execute the graph with tracing

# COMMAND ----------

# MAGIC %md
# MAGIC 5.1 Create a convenient function to chat with the LLM graph

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 5.2 Setup some messages

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>