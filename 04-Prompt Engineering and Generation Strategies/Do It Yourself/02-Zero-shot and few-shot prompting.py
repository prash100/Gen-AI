# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ####1. Using Transformer LLMs for Few Shot Learning

# COMMAND ----------

# MAGIC %md
# MAGIC 1.1 Create a pipeline for text-generation using a model Qwen/Qwen2.5-1.5B-Instruct

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 1.2 Try a Zero-shot generation for the given requirement.

# COMMAND ----------

instruction = """Given a word describing how someone is feeling, suggest a description of that person. 
The description should not include the original word."""
io_string = """
word: confused
description:"""


# COMMAND ----------

# MAGIC %md
# MAGIC 1.3 Generate the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 1.4 Create a few-shot prompt for the given requirement.

# COMMAND ----------

examples="""
word: happy
description: smiling, laughing, clapping

word: nervous
description: glancing around quickly, sweating, fidgeting

word: sleepy
description: heavy-lidded, slumping, rubbing eyes
"""


# COMMAND ----------

# MAGIC %md
# MAGIC 1.5 Generate the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 1.6 Improve the results of few-shot generation with the help of stop sequence

# COMMAND ----------

examples="""
word: happy
description: smiling, laughing, clapping
###
word: nervous
description: glancing around quickly, sweating, fidgeting
###
word: sleepy
description: heavy-lidded, slumping, rubbing eyes
###
"""


# COMMAND ----------

# MAGIC %md
# MAGIC 1.7 Generate the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ###2. Demostrate few shot to generate book summary
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Lets start with the zero-shot generation

# COMMAND ----------

prompt="""
Generate a book summary from the title:

book title: "Dune by Frank Herbert"
book description: """


# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Improve the results with few-shot to generate book summary

# COMMAND ----------

prompt="""
Generate a book summary from the title:

book title: "Stranger in a Strange Land by Robert A. Heinlein"
book description: "This novel tells the story of Valentine Michael Smith, a human who comes to Earth in early adulthood after being born on the planet Mars and raised by Martians, and explores his interaction with and eventual transformation of Terran culture."
###
book title: "The Adventures of Tom Sawyer by Mark Twain"
book description: "This novel is about a boy growing up along the Mississippi River. It is set in the 1840s in the town of St. Petersburg, which is based on Hannibal, Missouri, where Twain lived as a boy. In the novel, Tom Sawyer has several adventures, often with his friend Huckleberry Finn."
###
book title: "Dune by Frank Herbert"
book description: """

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>