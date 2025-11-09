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
# MAGIC 2.1 Create following data structures
# MAGIC 1. A data model for an structured LLM to produce SQL expression
# MAGIC 2. A data structure for passing state across graph nodes

# COMMAND ----------

from pydantic import BaseModel, Field

class SQL(BaseModel):
    "SQL Statement for the user question"
    sql_query: str = Field(description="An executable SQL expression to answer the user question")
    

# COMMAND ----------

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    #Message list
    messages: Annotated[list, "List of all LLM Messages", add_messages]
    #Input
    user_query: str
    #Output
    sql_query: str
    sql_explanation: str


# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Create following prompts
# MAGIC 1. A prompt for generating SQL expression from user question
# MAGIC 2. A prompt for explaining generated sql expression

# COMMAND ----------

from langchain_core.messages import SystemMessage

generate_instruction = SystemMessage("""You are a helpful data analyst who generates SQL queries for users based on their questions. 
       Assume you have a table named sales with columns product_id, product_name, and sales_amount. 
       Generate SQL query for the user question using this sales table""")

explain_instruction = SystemMessage("You are a helpful data analyst who briefly explains SQL queries to users.")       

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Create following LLM models
# MAGIC 1. A model to generate SQL expression
# MAGIC 2. A model to generate SQL explanation

# COMMAND ----------

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
sql_llm = llm.with_structured_output(SQL)

explain_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=500)

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Write a graph node function for SQL generation

# COMMAND ----------

def generate_sql(state: State) -> State:
    from langchain_core.messages import HumanMessage, AIMessage
    user_query = HumanMessage(state["user_query"])
    result = sql_llm.invoke([generate_instruction, user_query])
    return {"sql_query": result.sql_query, 
            "messages": [generate_instruction, user_query, AIMessage(result.sql_query)]}

# COMMAND ----------

# MAGIC %md
# MAGIC 2.5 Write a graph node function to generate SQL explanation

# COMMAND ----------

def explain_sql(state: State) -> State:
    from langchain_core.messages import HumanMessage, AIMessage
    sql_query = HumanMessage(state["sql_query"])
    result = explain_llm.invoke([explain_instruction, sql_query])
    return {"sql_explanation": result.content, 
            "messages": [explain_instruction, sql_query, result]}


# COMMAND ----------

# MAGIC %md
# MAGIC 2.6 Create a the graph with nodes and edges

# COMMAND ----------

from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("generate_sql", generate_sql)
builder.add_node("explain_sql", explain_sql)

builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "explain_sql")
builder.add_edge("explain_sql", END)

text_to_sql_app = builder.compile()

show_app_graph(text_to_sql_app)

# COMMAND ----------

# MAGIC %md
# MAGIC 2.7 Execute the graph with tracing

# COMMAND ----------

from langsmith import trace

question = "What is the total sales for each product?"
with trace(project_name="Text to SQL", 
           inputs={"question":question },
           name="Graph for text to sql") as rt:
    result = text_to_sql_app.invoke({"user_query":question})
    rt.end(outputs={"output":result})

# COMMAND ----------

# MAGIC %md
# MAGIC 2.8 Print the result

# COMMAND ----------

# MAGIC %md
# MAGIC 2.8.1 Print the sql query

# COMMAND ----------

print(result["sql_query"])

# COMMAND ----------

# MAGIC %md
# MAGIC 2.8.2 Print the sql explanation

# COMMAND ----------

print(result["sql_explanation"])

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>