# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ###Using Prompts - Example Prompts
# MAGIC 1. Prepare a text2text-generation pipeline for google/flan-t5-base
# MAGIC 2. Extract tokenizer and model
# MAGIC 3. Use the pipeline for the prompt based text generation

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####1. Translation Prompt

# COMMAND ----------

# MAGIC %md
# MAGIC 1.1 Create a prompt for translating the given text from english to french.

# COMMAND ----------

input_text = "I'm very happy to see you"


# COMMAND ----------

# MAGIC %md
# MAGIC 1.2 Generate using the pipeline

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 1.2. Use the generate method of the model

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####2. Summarization Prompt

# COMMAND ----------

# MAGIC %md
# MAGIC 2.1 Create a summarization prompt for the given article.

# COMMAND ----------

article = """India is a land with a vast variety of wildlife and a large variety of cultures. Situated in South Asia’s heartland, India is a densely populated country. It is a vastly diverse country in terms of culture, climate, religion, and language. India has chosen a number of emblems to represent our country’s image. Saffron, white, and green make up the Indian national flag. The Ashok chakra in the centre has a navy blue 24-spoke wheel that represents virtue. 
India is well-known for possessing the world’s greatest cultural diversity. Even for Indians, visiting and exploring every culture in India is quite difficult. India’s various cultures attract visitors from all over the world who want to come here at least once in their lives to experience India’s rich diversity.
India is a secular and democratic country that gives the liberty to practise any religion. Along with that, every individual in India has the liberty to read any religious book of their choice. Every individual has the liberty to move to any part of the country and adapt to the culture of that region. Every state of India has its own official language.
Jana Gana Mana is our national anthem, while Vande Matram is our national song. In the ‘Lion Capital of Asoka’, India’s national emblem, four lions sit back to back on a cylindrical base with four Ashok chakras on each side, only one of which is visible in the front. There are three lions visible and one concealed. It is a sign of sovereignty that also represents strength and bravery. It is a beautiful country that excels in art, culture, architecture, education, etc."""



# COMMAND ----------

# MAGIC %md
# MAGIC 2.2 Generate the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ####3. Sentiment Analysis Prompt

# COMMAND ----------

# MAGIC %md
# MAGIC 3.1 Create a sentiment analysis prompt for the given tweet.

# COMMAND ----------

tweet = """This movie is definitely one of my favorite movies of its kind. 
The interaction between respectable and morally strong characters is an ode 
to chivalry and the honor code amongst thieves and policemen."""

     

# COMMAND ----------

# MAGIC %md
# MAGIC 3.2 Generate result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####4. Question Answering Prompt

# COMMAND ----------

# MAGIC %md
# MAGIC 4.1 Create a prompt to answer the given question using the provided context.

# COMMAND ----------

context = """Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors."""
question = "What modern tool is used to make gazpacho?"

        

# COMMAND ----------

# MAGIC %md
# MAGIC 4.2 Generate the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####5. NER Prompt

# COMMAND ----------

# MAGIC %md
# MAGIC 5.1 Create a prompt for named entity recognition from the given text.

# COMMAND ----------

text = "The Golden State Warriors are an American professional basketball team based in San Francisco"



# COMMAND ----------

# MAGIC %md
# MAGIC 5.2 Generate the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>