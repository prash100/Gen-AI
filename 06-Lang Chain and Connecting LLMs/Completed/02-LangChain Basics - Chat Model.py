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

os.environ["OPENAI_API_KEY"]="Your OpenAI API Key>"

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Create and run Chat model

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Run with simple prompt\
# MAGIC Show an example to use gpt-4o-mini model

# COMMAND ----------

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=100)
output = llm.invoke(
    """You are a helpful assistant that generates multiple sub-questions related to an input question.
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
    Generate three sub-questions related to the given question.
    Question: How to solve data skew problem in Apache Spark?
    """)

# COMMAND ----------

output

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Run with role based prompt

# COMMAND ----------

from langchain_core.messages import SystemMessage, HumanMessage

system_msg = SystemMessage(
    """You are a helpful assistant that generates multiple sub-questions related to an input question.
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
    Generate three sub-questions related to the given question."""
)

human_msg = HumanMessage(
    "Question: How to solve data skew problem in Apache Spark?"
)

prompt = [system_msg, human_msg]

ai_message = llm.invoke(prompt)

# COMMAND ----------

ai_message.content

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Run with Chat Prompt Template

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [("system", """You are a helpful assistant that generates multiple sub-questions related to an input question.
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
    Generate three sub-questions related to the given question."""),
    ("human", "Question: {question}")]
)

prompt_value = prompt.invoke({"question": "How to solve data skew problem in Apache Spark?"})

ai_message = llm.invoke(prompt_value)

# COMMAND ----------

print(ai_message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Chain the prompt with LLM

# COMMAND ----------

llm_chain = prompt | llm
ai_message = llm_chain.invoke({"question": "How to solve data skew problem in Apache Spark?"})
print(ai_message.content)

# COMMAND ----------

ai_message

# COMMAND ----------

# MAGIC %md
# MAGIC 2.5 Generate Structured output

# COMMAND ----------

from pydantic import BaseModel, Field

class LLMResponse(BaseModel):
    "List of sub-questions along with the original question"
    question: str = Field(description="The question asked by the user")
    sub_questions: list[str] = Field(description = "A list of three sub-questions related to the given question")

llm_with_structured_output = llm.with_structured_output(LLMResponse)    

llm_chain = prompt | llm_with_structured_output

ai_message = llm_chain.invoke({"question": "How to solve data skew problem in Apache Spark?"})

# COMMAND ----------

ai_message.sub_questions[2]

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>