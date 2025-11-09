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
# MAGIC #### 2. Load the data and prepare for embedding

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Load data from the source file 

# COMMAND ----------

from langchain_community.document_loaders import TextLoader
loader = TextLoader("/Volumes/dev/genai_db/raw_data/test_data/markdown_sample.md")
text_doc = loader.load()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Split the document in smaller chunks

# COMMAND ----------

from langchain_text_splitters import MarkdownHeaderTextSplitter
headers_to_split_on = [
    ("#", "Heading_1"),
    ("##", "Heading_2")
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
doc_splits = markdown_splitter.split_text(text_doc[0].page_content)
#text_chunks = [split.page_content for split in doc_splits]

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Prepare and test the embedding model.\
# MAGIC We want to use the BAAI/bge-small-en-v1.5 model for embeddings.\
# MAGIC Create a function to return embeddings of a given chunk. We want to use this function later.

# COMMAND ----------

#from sentence_transformers import SentenceTransformer
#embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

#def embed_chunk(chunk):
#    return embedding_model.encode([chunk], normalize_embeddings=True).tolist()[0]

from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                        cache_folder = "/Volumes/dev/genai_db/models/bge-small-en-v1.5/")

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Find the number of embedding dimensions for later usage.

# COMMAND ----------

#test_embedding = embed_chunk(text_chunks[0])
#embedding_dim = len(test_embedding)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3 Prepare the Vector Database
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Create Vector Database. We want to use pymilvus vector database.

# COMMAND ----------

#from pymilvus import MilvusClient
#milvius_client = MilvusClient("/tmp/milvus_demo.db")

from langchain_milvus import Milvus

db_location = "/tmp/milvus_demo.db"
collection_name = "rag_collection"

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Create a collection for your embeddings

# COMMAND ----------

'''
collection_name = "rag_collection"

if milvius_client.has_collection(collection_name):
    milvius_client.drop_collection(collection_name)

milvius_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",
    consistency_level="Strong"
)
'''

vector_db = Milvus(
    embedding_function=embedding_model,
    connection_args={"uri": db_location},
    collection_name=collection_name,
    drop_old=True,
    auto_id=True,
    consistency_level="Strong",
    index_params={"metric_type": "IP"}
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Prepare and load your embedding collection

# COMMAND ----------

# MAGIC %md
# MAGIC 4.1 Prepare an embedding table

# COMMAND ----------

'''
chunks_table = []

for i, chunk in enumerate(text_chunks):
    chunks_table.append({"id": i, "vector": embed_chunk(chunk), "text": chunk})

'''

# COMMAND ----------

# MAGIC %md
# MAGIC 4.2 Load embedding table into collection

# COMMAND ----------

#insert_result = milvius_client.insert(collection_name=collection_name, data=chunks_table)
#insert_result["insert_count"]

vector_db.add_documents(documents=doc_splits)

# COMMAND ----------

# MAGIC %md
# MAGIC ####5. Build the RAG pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC 5.1 Load the generation model pipeline. We want to use meta-llama/Llama-3.2-1B-Instruct model for generation

# COMMAND ----------

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

pipe = pipeline("text-generation", 
                model="meta-llama/Llama-3.2-1B-Instruct",
                max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=pipe)

# COMMAND ----------

# MAGIC %md
# MAGIC 5.2 Get your question.

# COMMAND ----------

question1 = "What is AQE?"
question2 = "How to check whether AQE is enabled?"
question3 = "How AQE optimizes Spark SQL?"
question4 = "How Caching optimizes Spark SQL?"

# COMMAND ----------

# MAGIC %md
# MAGIC 5.3 Search related contexts for the question in the vector database

# COMMAND ----------

'''
search_result = milvius_client.search(
    collection_name=collection_name,
    data=[embed_chunk(question4)],
    limit=1,
    search_params={"metric_type": "IP"},
    output_fields=["text"]
)
'''
retriever = vector_db.as_retriever(search_kwargs={"k": 1})

# COMMAND ----------

search_result = retriever.invoke(question1)

# COMMAND ----------

# MAGIC %md
# MAGIC 5.4 Prepare the context

# COMMAND ----------

#context = search_result[0][0]["entity"]["text"]

# COMMAND ----------

# MAGIC %md
# MAGIC 5.5 Create an augumented prompt to answer the given question using the retrived context.

# COMMAND ----------

'''
prompt = "Answer the question using the given context\n" + \
    "Context: " + context + "\n" + \
    "Question: " + question4 + "\n" + \
    "Answer: "
'''
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("""Answer the question using the given context
                                      Context: {context}
                                      Question: {question}
                                      Answer:""")


# COMMAND ----------

# MAGIC %md
# MAGIC 5.5.2 Test your prompt

# COMMAND ----------

prompt.invoke({"context": retriever, "question": question1})

# COMMAND ----------

# MAGIC %md
# MAGIC 5.6 Generate the result

# COMMAND ----------

#result = pipe(prompt, max_new_tokens=100)
#print(result[0]["generated_text"])

# COMMAND ----------

# MAGIC %md
# MAGIC 5.6 Create a RAG chain

# COMMAND ----------

# MAGIC %md
# MAGIC 5.6.1 Create a function for combining multiple splits

# COMMAND ----------

def join_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# COMMAND ----------

# MAGIC %md
# MAGIC 5.6.2 Create a chain

# COMMAND ----------

#Input = question
#Potential Chain: 
# rag_docs = retriever(question) 
#   -> context = join_docs(rag_docs) 

#       -> prompt_value = prompt(context, question) 
#           -> result = llm(prompt_value) 
#               -> answer = parse_output(result)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = ({"context" : RunnablePassthrough() | retriever | join_docs,
             "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser())


# COMMAND ----------

# MAGIC %md
# MAGIC 5.6.3 Execute the RAG chain and get the result

# COMMAND ----------

result = rag_chain.invoke(question1)

# COMMAND ----------

result

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>