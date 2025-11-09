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

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    #Message list
    messages: Annotated[list, add_messages]

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Create necessory tools

# COMMAND ----------

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

search = DuckDuckGoSearchRun()

@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return eval(query)

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Create an LLM to answer user questions and bind it with tools

# COMMAND ----------

from langchain_openai import ChatOpenAI

tools = [search, calculator]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Write a graph node function to answer user questions

# COMMAND ----------

def chatbot(state: State) -> State:
    answer = llm.invoke(state["messages"])
    return {"messages": answer}

# COMMAND ----------

# MAGIC %md
# MAGIC 2.5 Create a the graph with nodes and edges

# COMMAND ----------

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

chatbot_app = builder.compile(checkpointer=MemorySaver())
show_app_graph(chatbot_app)


# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Execute the graph with tracing

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Create a convenient function to chat with the LLM graph

# COMMAND ----------


from langsmith import trace
from langchain_core.messages import HumanMessage

def chat(thread_id, question):
    with trace(project_name="ReAct Bot",
               inputs={"question": question},
               name="Chat",
               metadata = {"thread_id": thread_id}) as rt:
        chat_thread = {"configurable": {"thread_id": thread_id}}
        answer = chatbot_app.invoke({"messages": [HumanMessage(question)]}, chat_thread)
        rt.end(outputs={"output": answer})


# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Setup some messages

# COMMAND ----------

question_1 = "Who is the president of india as on April 2025?"
question_2 = "What is her age as on the year 2025?"

thread_id = "President_of_india_102"
chat(thread_id, question_1)
chat(thread_id, question_2)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>