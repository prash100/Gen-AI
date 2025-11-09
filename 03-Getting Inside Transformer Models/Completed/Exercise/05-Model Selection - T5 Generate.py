# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ####Using Transformer LLMs for Text Summarization

# COMMAND ----------

article_india = """summarize: India is a land with a vast variety of wildlife and a large variety of cultures. Situated in South Asia’s heartland, India is a densely populated country. It is a vastly diverse country in terms of culture, climate, religion, and language. India has chosen a number of emblems to represent our country’s image. Saffron, white, and green make up the Indian national flag. The Ashok chakra in the centre has a navy blue 24-spoke wheel that represents virtue. 
India is well-known for possessing the world’s greatest cultural diversity. Even for Indians, visiting and exploring every culture in India is quite difficult. India’s various cultures attract visitors from all over the world who want to come here at least once in their lives to experience India’s rich diversity.
India is a secular and democratic country that gives the liberty to practise any religion. Along with that, every individual in India has the liberty to read any religious book of their choice. Every individual has the liberty to move to any part of the country and adapt to the culture of that region. Every state of India has its own official language.
Jana Gana Mana is our national anthem, while Vande Matram is our national song. In the ‘Lion Capital of Asoka’, India’s national emblem, four lions sit back to back on a cylindrical base with four Ashok chakras on each side, only one of which is visible in the front. There are three lions visible and one concealed. It is a sign of sovereignty that also represents strength and bravery. It is a beautiful country that excels in art, culture, architecture, education, etc."""

# COMMAND ----------

article_america = """summarize: America has changed dramatically during recent years. Not only has the number of graduates in traditional engineering disciplines such as mechanical, civil, electrical, chemical, and aeronautical engineering declined, but in most of the premier American universities engineering curricula now concentrate on and encourage largely the study of engineering science. As a result, there 
are declining offerings in engineering subjects dealing with infrastructure, the environment, and related issues, and greater concentration on high technology subjects, largely supporting increasingly complex scientific developments. While the latter is important, it should not be at the expense of more traditional engineering.
Rapidly developing economies such as China and India, as well as other industrial countries in Europe and Asia, continue to encourage and advance the teaching of engineering. Both China and India, respectively, graduate six and eight times as many traditional engineers as does the United States. 
Other industrial countries at minimum maintain their output, while America suffers an increasingly serious decline in the number of engineering graduates and a lack of well-educated engineers."""

# COMMAND ----------

# MAGIC %md
# MAGIC Q1: Assuming you have selected T5 for the text summarization. Perform text summarization on given customer comments.
# MAGIC 1. Do not use pipeline function
# MAGIC 2. Use Model Classes
# MAGIC 3. Apply t5-small model from huggingface hub

# COMMAND ----------

# MAGIC %md
# MAGIC Step 1 - Create a tokenizer and prepare input tokens

# COMMAND ----------

from transformers import T5Tokenizer

checkpoint = "T5-small"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)

inputs = tokenizer([article_india, article_america], padding=True, return_tensors="pt")
print(inputs)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 2 - Create an use a model to generate raw outputs
# MAGIC

# COMMAND ----------

from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained(checkpoint)
raw_outputs = model.generate(inputs.input_ids, min_length=20, max_length=50)
print(raw_outputs)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 3 - Decode raw outputs to summary text

# COMMAND ----------

tokenizer.batch_decode(raw_outputs, skip_special_tokens=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Q2: Assuming you have selected T5 for the text summarization. Perform text summarization on given customer comments.
# MAGIC 1. Do not use pipeline function
# MAGIC 2. Use Auto Classes
# MAGIC 3. Apply T5-small model from huggingface hub

# COMMAND ----------

# MAGIC %md
# MAGIC Step 1 - Create a tokenizer and prepare input tokens

# COMMAND ----------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

inputs = tokenizer([article_india, article_america], padding=True, return_tensors="pt")
print(inputs)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 2 - Create an use a model to generate raw outputs

# COMMAND ----------

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
raw_outputs = model.generate(inputs.input_ids, min_length=20, max_length=50)
print(raw_outputs)

# COMMAND ----------

# MAGIC %md
# MAGIC Step 3 - Decode raw outputs to summary text

# COMMAND ----------

tokenizer.batch_decode(raw_outputs, skip_special_tokens=True)

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>