# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Complex NLP Tasks
# MAGIC 1. Token Classification
# MAGIC 2. Text Classification
# MAGIC 3. Text Generation

# COMMAND ----------

# MAGIC %md
# MAGIC ###1. Token Classification
# MAGIC Token classification is a natural language understanding task in which a label is assigned to some tokens in a text. Some popular token classification subtasks are listed below.
# MAGIC 1. <b>Named Entity Recognition (NER) - </b> NER models could be trained to identify specific entities in a text, such as entities, individuals and places.
# MAGIC 2. <b>Part-of-Speech (PoS) tagging - </b> PoS tagging would identify, for example, which words in a text are verbs, nouns, and punctuation marks.

# COMMAND ----------

from transformers import pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC #####1.1 Named Entity Recognition 
# MAGIC NER models could be trained to identify specific entities in a text, such as entities, individuals and places.
# MAGIC
# MAGIC <br>
# MAGIC <img src ='https://learningjournal.github.io/pub-resources/images/ner.jpg' alt="NER" style="width: 300px">

# COMMAND ----------

# MAGIC %md
# MAGIC Q1.1 Execute a named entity recognition task on the given string.

# COMMAND ----------

text = "Hello I'm Omar and I live in Zurich."

# COMMAND ----------

# MAGIC %md
# MAGIC #####1.2 Part-of-Speech (PoS) Tagging
# MAGIC The model recognizes parts of speech, such as nouns, pronouns, adjectives, or verbs, in a given text.
# MAGIC
# MAGIC <br>
# MAGIC <img src ='https://learningjournal.github.io/pub-resources/images/pos.jpg' alt="POS" style="width: 300px">

# COMMAND ----------

# MAGIC %md
# MAGIC Q1.2 Execute Parts of Speech tagging over the given string.\
# MAGIC You can use the vblagoje/bert-english-uncased-finetuned-pos model.

# COMMAND ----------

sentance = "My Name is Thomas and I work at Hugging Face in Brooklyn."

# COMMAND ----------

# MAGIC %md
# MAGIC ###2. Text Classification
# MAGIC Text Classification is the task of assigning a label or class to a given text. Some common subtasks are listed below.
# MAGIC 1. Natural Language Inference (NLI) 
# MAGIC 2. Question Natural Language Inference (QNLI)
# MAGIC 3. Sentiment Analysis
# MAGIC 4. Quora Question Pairs
# MAGIC 5. Grammatical Correctness

# COMMAND ----------

# MAGIC %md
# MAGIC #####2.1 Natural Language Inference (NLI) 
# MAGIC The model determines the relationship between two given texts. Precisely, the model takes a premise and a hypothesis and returns a class that can either be:
# MAGIC 1. entailment, which means the hypothesis is true.
# MAGIC 2. contraction, which means the hypothesis is false.
# MAGIC 3. neutral, which means there's no relation between the hypothesis and the premise.

# COMMAND ----------

# MAGIC %md
# MAGIC Q2.1 Execute NLI for the given premise and Hypothesis.\
# MAGIC You can use the roberta-large-mnli model.

# COMMAND ----------

prompt = """Premise: Soccer game with multiple males playing.
            Hypothesis: Some men are playing a sport."""        

# COMMAND ----------

# MAGIC %md
# MAGIC #####2.2 Question Natural Language Inference (QNLI)
# MAGIC QNLI is the task of determining if the answer to a certain question can be found in a given document. If the answer can be found the label is “entailment”. If the answer cannot be found the label is “not entailment".

# COMMAND ----------

# MAGIC %md
# MAGIC Q2.2 Execute a QNLI task for the given question and answer.\
# MAGIC You can use the cross-encoder/qnli-electra-base model.

# COMMAND ----------

prompt = """Question: Where is the capital of France? 
            Sentence: Paris is the capital of Japan."""       

# COMMAND ----------

# MAGIC %md
# MAGIC #####2.3 Sentiment Analysis
# MAGIC Determine the class which could be polarities like positive, negative, neutral, or sentiments such as happiness or anger.

# COMMAND ----------

# MAGIC %md
# MAGIC Q2.3 Perform sentiment analysis for the given statement.

# COMMAND ----------

statement = "I loved Star Wars so much!"

# COMMAND ----------

# MAGIC %md
# MAGIC #####2.4 Quora Question Pairs
# MAGIC Assess whethe two provided questions are paraphrases of each other. 

# COMMAND ----------

# MAGIC %md
# MAGIC Q2.4 Perform QQP analysis on the given questions.\
# MAGIC You can use the textattack/bert-base-uncased-QQP model.

# COMMAND ----------

questions = "Which city is the capital of France?, Where is the capital of France?"

# COMMAND ----------

# MAGIC %md
# MAGIC #####2.5 Grammatical Correctness
# MAGIC Assessing the grammatical acceptability of a sentence. The classes in this task are “acceptable” and “unacceptable”. 

# COMMAND ----------

# MAGIC %md
# MAGIC Q2.5 Execute Corpus of Linguistic Acceptability (CoLA) analysis for the given sentance.\
# MAGIC You can use the textattack/distilbert-base-uncased-CoLA model.

# COMMAND ----------

statement = "I will walk to home when I went through the bus."

# COMMAND ----------

# MAGIC %md
# MAGIC ###3. Text Generation
# MAGIC Generating text is the task of generating new text given another text. These models can, for example, fill in incomplete text or paraphrase.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3.1 Text Generation

# COMMAND ----------

# MAGIC %md
# MAGIC Q3.1 Generate some text based on the given text.\
# MAGIC The maximum length of the text should be 30 words and show three possible options.

# COMMAND ----------

text = "Hello, I'm a language model"

# COMMAND ----------

# MAGIC %md
# MAGIC #####3.2 Text to Text Generation

# COMMAND ----------

# MAGIC %md
# MAGIC Q3.2 Answer the given question using the provided context.

# COMMAND ----------

prompt = "question: What is 42 ? context: 42 is the answer to life, the universe and everything"

# COMMAND ----------

# MAGIC %md
# MAGIC Q3.3 Translate the given sentance from english to french using a text to text generation model.

# COMMAND ----------

prompt = "translate from English to Italian: I'm very happy"

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>