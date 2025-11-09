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
# MAGIC 1.2 Get your questions ready

# COMMAND ----------

question_py = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

question_js = """Why doesn't the following code work:

import { ChatPromptTemplate } from "@langchain/core/prompts";

const chatPrompt = ChatPromptTemplate.fromMessages([
  ["human", "speak in {language}"],
]);

const formattedChatPrompt = await chatPrompt.invoke({
  input_language: "french"
});
"""

# COMMAND ----------

# MAGIC %md 
# MAGIC 1.3 Define a function to show the graph

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
# MAGIC #### 2. Implement RAG Data Prepration
# MAGIC Create a class for the following functionality
# MAGIC 1. Load data from PDF source
# MAGIC 2. Split documents into smaller chunks
# MAGIC 3. Connect to vector DB and store document chunks
# MAGIC 4. Create a vector DB retriever

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Implement RAG Functionality
# MAGIC 1. Create a data model for structured llm output
# MAGIC 2. Create a data model to track the state across nodes

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Create a data model for structured llm output

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Create a data model to track the state across nodes

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.3 Create a RagBot class for the following functionality
# MAGIC 1. Create a router node
# MAGIC 2. Create a retrieve_py_docs node
# MAGIC 3. Create a retrieve_js_docs node
# MAGIC 4. Create a select_retriever function for condition evaluation
# MAGIC 5. Create a generate_answer node
# MAGIC 6. Create a function to build the state graph

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.4 Create your ragbot_app and display the graph

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.5 Execute and test your ragbot_app for a python problem

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3.6 Execute and test your ragbot_app for a js problem

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>