# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ####1. Setup your Environment Variables
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC No API Key Required

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Using Document Loaders

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Load data from text file

# COMMAND ----------

from langchain_community.document_loaders import TextLoader

loader = TextLoader(file_path="/Volumes/dev/genai_db/raw_data/test_data/article.txt", autodetect_encoding=True)
text_doc = loader.load()

# COMMAND ----------

text_doc[0].page_content

# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Load data from CSV file

# COMMAND ----------

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='/Volumes/dev/genai_db/raw_data/test_data/invoices.csv', autodetect_encoding=True,
                   csv_args={
                       'delimiter': ',',
                       'quotechar': '"'
                   }
                   )

csv_doc = loader.load()

# COMMAND ----------

csv_doc

# COMMAND ----------

# MAGIC %md
# MAGIC 2.3 Load data from web

# COMMAND ----------

from langchain_community.document_loaders import WebBaseLoader
import bs4 #For documentation refer to https://www.crummy.com/software/BeautifulSoup/bs4/doc/#

loader = WebBaseLoader(web_path="https://spark.apache.org/docs/latest/sql-performance-tuning.html",
                       bs_kwargs={"parse_only": bs4.SoupStrainer(id=("content"))})

web_doc = loader.load()

# COMMAND ----------

web_doc

# COMMAND ----------

# MAGIC %md
# MAGIC 2.4 Load data from PDF

# COMMAND ----------

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="/Volumes/dev/genai_db/raw_data/test_data/book.pdf")

pdf_doc = loader.load()

# COMMAND ----------

pdf_doc

# COMMAND ----------

# MAGIC %md
# MAGIC 2.5 Splitting large document

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=300, 
                                          separators=["\n\n\n\n", "\n\n\n", "\n\n", "\n"] )

splitted_doc = splitter.split_documents(text_doc)

# COMMAND ----------

len(splitted_doc)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>