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

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, START, StateGraph
from langchain_experimental.utilities.python import PythonREPL

#Define state
class CoderState(TypedDict):
    messages: Annotated[list, add_messages]

#Define Python repl tool
@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):

    """Use this to execute python code and and display chart. This chart is visible to the user."""
    repl = PythonREPL()
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return result_str

#Create llm
coder_llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([python_repl_tool])

#Define coder node function
def coder(state: CoderState) -> CoderState:
    answer = coder_llm.invoke(state["messages"])
    return {"messages": answer}

#Create coder agent app
builder = StateGraph(CoderState)
builder.add_node("coder", coder)
builder.add_node("tools", ToolNode([python_repl_tool]))

builder.add_edge(START, "coder")
builder.add_conditional_edges("coder", tools_condition)
builder.add_edge("tools", "coder")

coder_app = builder.compile()
show_app_graph(coder_app)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Create Researcher Agent

# COMMAND ----------

from langchain_community.tools import DuckDuckGoSearchRun

#Define state
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]

#Define search tool
search = DuckDuckGoSearchRun()

#Create llm
search_llm = ChatOpenAI(model="gpt-4o-mini").bind_tools([search])

#Define research node function
def research(state: ResearchState) -> ResearchState:
    result = search_llm.invoke(state["messages"])
    return {"messages": result}

#Create research agent app
builder = StateGraph(ResearchState)
builder.add_node("research", research)
builder.add_node("tools", ToolNode([search]))

builder.add_edge(START, "research")
builder.add_conditional_edges("research", tools_condition)
builder.add_edge("tools", "research")

research_app = builder.compile()
show_app_graph(research_app)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Prepare for Supervisor
# MAGIC Note: Supervisor implements routing amongst available agents (researcher, coder).

# COMMAND ----------

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from typing import Literal

#Define structured output data model
class Route(BaseModel):
    """Worker to route to next node. If no workers needed, route to FINISH."""
    next_node: Literal["researcher", "coder", "FINISH"] = Field(description="Worker to route to next node. If no workers needed, route to FINISH.")

#Create structured output model
supervisor_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(Route)

#Create supervisor prompt for routing
agents = ["researcher", "coder"]
supervisor_prompt = SystemMessage(
    f"""You are a supervisor tasked with managing a conversation between the following workers: {agents}. 
    Given the following user request, respond with the worker to act next. 
    Each worker will perform a task and respond with their results and status. 
    When finished, respond with FINISH."""
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ####4. Create supervisor architecture

# COMMAND ----------

# MAGIC %md
# MAGIC 4.1 Create required state and functions

# COMMAND ----------

from typing import Literal, TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage

#Define supervisor state
class State(TypedDict):
    #Message list
    messages: Annotated[list, add_messages]
    next_node: str

#Define supervisor node function
def supervisor(state: State) -> State:
    result = supervisor_llm.invoke([supervisor_prompt] + state["messages"])
    return {"messages": AIMessage(result.next_node), "next_node": result.next_node}

#Define routing function for conditional edge
def node_selector(state: State) -> Literal["researcher", "coder", "__end__"]:
        if state["next_node"] == "FINISH":
            return "__end__"
        else:
            return state["next_node"]

# COMMAND ----------

# MAGIC %md
# MAGIC 4.2 Create a the graph with nodes and edges

# COMMAND ----------

from langgraph.graph import END, START, StateGraph

builder = StateGraph(State)
builder.add_node("researcher", research_app)
builder.add_node("coder", coder_app)
builder.add_node("supervisor", supervisor)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", node_selector)
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

supervisor_app = builder.compile()
show_app_graph(supervisor_app)


# COMMAND ----------

# MAGIC %md
# MAGIC ####5. Execute the graph with tracing

# COMMAND ----------

# MAGIC %md
# MAGIC 5.1 Create a convenient function to chat with the LLM graph

# COMMAND ----------

from langsmith import trace
from langchain_core.messages import HumanMessage

def chat(question):
    with trace(project_name="MultiAgent",
               inputs={"question": question},
               name="chat") as rt:
        answer = supervisor_app.invoke({"messages": [HumanMessage(question)]})
        rt.end(outputs={"output": answer})

# COMMAND ----------

# MAGIC %md
# MAGIC 5.2 Setup some messages

# COMMAND ----------

request = "How is the GDP of Uttar Pradesh has grown in last 3 years from 2023. Show me a bar chart for the same"

chat(request)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>