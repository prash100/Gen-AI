# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ####Using Transformer LLMs for common tasks
# MAGIC 1. Summarization
# MAGIC 2. Translation
# MAGIC 3. Question Answering
# MAGIC 4. Table Question Answering
# MAGIC 5. Fill Mask
# MAGIC 6. Feature Extraction
# MAGIC 7. Zero Shot Classification

# COMMAND ----------

from transformers import pipeline
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ####1. Summarization
# MAGIC Summarization is the task of producing a shorter version of a document while preserving its important information. Some models can extract text from the original input, while other models can generate entirely new text.
# MAGIC
# MAGIC <br>
# MAGIC <img src ='https://learningjournal.github.io/pub-resources/images/summarization.jpg' alt="Summarization" style="width: 300px">

# COMMAND ----------

# MAGIC %md
# MAGIC Q1. Create a summary of the given article.

# COMMAND ----------

article = """Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles). The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017."""

# COMMAND ----------

summarizer = pipeline(task="summarization", max_length=50)
result = summarizer(article)

# COMMAND ----------

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ####2. Translation
# MAGIC Translation is the task of converting text from one language to another.
# MAGIC <br>
# MAGIC <img src ='https://learningjournal.github.io/pub-resources/images/translation.jpg' alt="Translation" style="width: 300px">

# COMMAND ----------

# MAGIC %md
# MAGIC Q2. Translate the given statement from english to french.\
# MAGIC You can use the Helsinki-NLP/opus-mt-en-fr model

# COMMAND ----------

statement = "My name is Prashant."
translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-fr")
result = translator(statement)

# COMMAND ----------

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Question Answering
# MAGIC Question Answering models can retrieve the answer to a question from a given text, which is useful for searching for an answer in a document. Some question answering models can generate answers without context!
# MAGIC <br>
# MAGIC <img src ='https://learningjournal.github.io/pub-resources/images/question_answer.jpg' alt="Question Answer" style="width: 300px">

# COMMAND ----------

# MAGIC %md
# MAGIC Q3. Generate the answer for the given question from the context.

# COMMAND ----------

question = "Where do I live?"
context = "My name is Merve and I live in İstanbul."
qa_pipeline = pipeline(task="question-answering")
result = qa_pipeline(question = question, context = context)

# COMMAND ----------

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ####4. Table Question Answering
# MAGIC Table Question Answering (Table QA) is the answering a question about an information on a given table.
# MAGIC
# MAGIC <br>
# MAGIC <img src ='https://learningjournal.github.io/pub-resources/images/table_question_answer.jpg' alt="Table Question Answer" style="width: 300px">

# COMMAND ----------

# MAGIC %md
# MAGIC Q4. Generate the answer for the given question from the provided table.\
# MAGIC You can use the google/tapas-large-finetuned-wtq model

# COMMAND ----------

question = "how many movies does Leonardo Di Caprio have?"
data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
table = pd.DataFrame.from_dict(data)

tqa_pipeline = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")
result = tqa_pipeline(table=table, query=question)

# COMMAND ----------

print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Fill-Mask
# MAGIC Masked language modeling is the task of masking some of the words in a sentence and predicting which words should replace those masks. These models are useful when we want to get a statistical understanding of the language in which the model is trained in.
# MAGIC
# MAGIC <br>
# MAGIC <img src ='https://learningjournal.github.io/pub-resources/images/fill_mask.jpg' alt="Fill Mask" style="width: 300px">

# COMMAND ----------

# MAGIC %md
# MAGIC Q5. Generate the sentance after filling the missing word.

# COMMAND ----------

sentance = "Paris is the <mask> of France."
fm_pipeline = pipeline(task="fill-mask")
result = fm_pipeline(sentance)

# COMMAND ----------

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ####6. Feature Extraction
# MAGIC Feature extraction is the task of extracting features learnt in a model. These models can be used in RAG Approch.
# MAGIC
# MAGIC <br>
# MAGIC <img src ='https://learningjournal.github.io/pub-resources/images/feature_extraction.jpg' alt="Feature Extraction" style="width: 300px">

# COMMAND ----------

# MAGIC %md
# MAGIC Q6. Extract the features of the given text.\
# MAGIC You can use the facebook/bart-base model.

# COMMAND ----------

text = "Transformers is an awesome library!"
feature_extractor = pipeline(task="feature-extraction", model="facebook/bart-base")
result = feature_extractor(text, return_tensors = "pt")

# COMMAND ----------

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ####7. Zero Shot Classification
# MAGIC Zero-shot text classification is a task in natural language processing where a model is trained on a set of labeled examples but is then able to classify new examples from previously unseen classes.
# MAGIC
# MAGIC <br>
# MAGIC <img src ='https://learningjournal.github.io/pub-resources/images/zero_shot_classification.jpg' alt="Zero Shot Classification" style="width: 300px">

# COMMAND ----------

# MAGIC %md
# MAGIC Q7. Classify the given sentance to the provided lables.\
# MAGIC You can use the facebook/bart-large-mnli model.

# COMMAND ----------

sentance = "I have a problem with my iphone that needs to be resolved asap!"
candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"]
zsc_pipiline = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")
result = zsc_pipiline(sentance, candidate_labels = candidate_labels)

# COMMAND ----------

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>