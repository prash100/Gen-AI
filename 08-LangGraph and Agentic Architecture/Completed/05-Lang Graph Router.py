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

class RagRetriever():
    def __init__(self, pdf_location, db_location):
        self.pdf_location = pdf_location
        self.db_location = db_location
        self.collection_name = "rag_collection"
        self.embedding_model_name = "BAAI/bge-small-en-v1.5"
    
    def load_pdf(self):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(self.pdf_location)
        return loader.load()
    
    def split_docs(self, docs):
        from langchain_text_splitters import RecursiveCharacterTextSplitter        
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, separators=["\n\n\n", "\n\n"])
        splitted_docs = splitter.split_documents(docs)        
        return splitted_docs
    
    def store_docs(self, docs):
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embedding_model = HuggingFaceEmbeddings(model=self.embedding_model_name,
                                                cache_folder = "/Volumes/dev/genai_db/models/bge-small-en-v1.5/")
        vector_db =  Chroma(
            embedding_function=embedding_model,
            collection_name=self.collection_name,
            persist_directory=self.db_location)
        vector_db.reset_collection()
        vector_db.add_documents(documents=docs)
        return vector_db  

    def get(self):
        docs = self.load_pdf()
        splitted_doc = self.split_docs(docs)
        vector_db = self.store_docs(splitted_doc)
        return vector_db.as_retriever(search_kwargs={"k": 2})

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Implement RAG Functionality
# MAGIC 1. Create a data model for structured llm output
# MAGIC 2. Create a data model to track the state across nodes

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Create a data model for structured llm output

# COMMAND ----------

from pydantic import BaseModel, Field
from typing import Literal

class Route(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasourse: Literal["python_docs", "js_docs"] = Field(
        description="Given a user question choose which datasource would be most relevant for answering their question"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Create a data model to track the state across nodes

# COMMAND ----------

from typing import TypedDict, Annotated
from langchain_core.documents import Document

class State(TypedDict):
    #Input
    user_query: str
    #Output
    route: str
    documents: list[Document]
    answer: str

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

class RagBot():
    def __init__(self):
        self.model_name = "gpt-4o-mini"
        self.py_doc_source = "/Volumes/dev/genai_db/raw_data/test_data/PromptTemplates_Python.pdf"
        self.py_db_location = "/tmp/python_docs.db"
        self.js_doc_source = "/Volumes/dev/genai_db/raw_data/test_data/PromptTemplates_JS.pdf"
        self.js_db_location = "/tmp/js_docs.db"

    def router(self, state: State) -> State:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from langsmith import Client

        llm = ChatOpenAI(model=self.model_name, temperature=0.1, max_tokens=50)
        router_llm = llm.with_structured_output(Route)
        smith = Client()
        prompt = smith.pull_prompt("sys_inst_route")
        chain = prompt | router_llm
        route = chain.invoke({"question": state["user_query"]})
        return {"route": route.datasourse.lower()}
    
    def retrieve_py_docs(self, state: State) -> State:
        retriever = RagRetriever(self.py_doc_source, self.py_db_location)
        py_retriever = retriever.get()
        docs = py_retriever.invoke(state["user_query"])
        return {"documents": docs} 
    
    def retrieve_js_docs(self, state: State) -> State:
        retriever = RagRetriever(self.js_doc_source, self.js_db_location)
        js_retriever = retriever.get()
        docs = js_retriever.invoke(state["user_query"])    
        return {"documents": docs}

    def select_retriever(self, state: State) -> Literal["retrieve_py_docs", "retrieve_js_docs"]:
        if state["route"] == "python_docs":
            return "retrieve_py_docs"
        else:
            return "retrieve_js_docs"

    def generate_answer(self, state: State) -> State:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from langsmith import Client

        llm = ChatOpenAI(model=self.model_name, temperature=0.7, max_tokens=500)    
        context = "\n".join(doc.page_content.strip() for doc in state["documents"])
        smith = Client()
        prompt = smith.pull_prompt("sys_inst_rag_answer")

        prompt_value = prompt.invoke({"context": context, "question": state["user_query"]})
        answer = llm.invoke(prompt_value)
        return {"answer": answer.content}
    
    def get_app(self):
        from langgraph.graph import END, START, StateGraph
        builder = StateGraph(State)
        builder.add_node("router", self.router)
        builder.add_node("retrieve_py_docs", self.retrieve_py_docs)
        builder.add_node("retrieve_js_docs", self.retrieve_js_docs)
        builder.add_node("generate_answer", self.generate_answer)

        builder.add_edge(START, "router")
        builder.add_conditional_edges("router", self.select_retriever)
        builder.add_edge("retrieve_py_docs", "generate_answer")
        builder.add_edge("retrieve_js_docs", "generate_answer")
        builder.add_edge("generate_answer", END)

        ragbot = builder.compile()
        return ragbot


# COMMAND ----------

# MAGIC %md
# MAGIC 3.4 Create your ragbot_app and display the graph

# COMMAND ----------

ragbot = RagBot()
ragbot_app = ragbot.get_app()
show_app_graph(ragbot_app)

# COMMAND ----------

# MAGIC %md
# MAGIC 3.5 Execute and test your ragbot_app for a python problem

# COMMAND ----------

from langsmith import trace

question = question_py
with trace(project_name="RAG Bot", 
           inputs={"question": question },
           name="Py RaGbot Graph") as rt:
    result = ragbot_app.invoke({"user_query": question})
    rt.end(outputs={"output": result["answer"]})

# COMMAND ----------

# MAGIC %md
# MAGIC 3.6 Execute and test your ragbot_app for a js problem

# COMMAND ----------

from langsmith import trace

question = question_js
with trace(project_name="RAG Bot", 
           inputs={"question": question },
           name="JS RaGbot Graph") as rt:
    result = ragbot_app.invoke({"user_query": question})
    rt.end(outputs={"output": result["answer"]})

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>