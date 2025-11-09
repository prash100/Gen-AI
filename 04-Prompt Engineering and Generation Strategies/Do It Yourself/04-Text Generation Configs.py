# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ###Text generation configs

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Prepare the environment for a latest model

# COMMAND ----------

pip install --upgrade transformers

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Download the model meta-llama/Llama-3.2-1B-Instruct and create tokenizer and model

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 3. Prepare your prompt for the summarization

# COMMAND ----------

prompt = """
Summarize the given article. Do not create half sentences in the summary.
Article: India is a land with a vast variety of wildlife and a large variety of cultures. Situated in South Asia’s heartland, India is a densely populated country. It is a vastly diverse country in terms of culture, climate, religion, and language. India has chosen a number of emblems to represent our country’s image. Saffron, white, and green make up the Indian national flag. The Ashok chakra in the centre has a navy blue 24-spoke wheel that represents virtue. 
India is well-known for possessing the world’s greatest cultural diversity. Even for Indians, visiting and exploring every culture in India is quite difficult. India’s various cultures attract visitors from all over the world who want to come here at least once in their lives to experience India’s rich diversity.
India is a secular and democratic country that gives the liberty to practise any religion. Along with that, every individual in India has the liberty to read any religious book of their choice. Every individual has the liberty to move to any part of the country and adapt to the culture of that region. Every state of India has its own official language.
Jana Gana Mana is our national anthem, while Vande Matram is our national song. In the ‘Lion Capital of Asoka’, India’s national emblem, four lions sit back to back on a cylindrical base with four Ashok chakras on each side, only one of which is visible in the front. There are three lions visible and one concealed. It is a sign of sovereignty that also represents strength and bravery. It is a beautiful country that excels in art, culture, architecture, education, etc.
Summary:
"""

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Generate two sequences

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 5. Decode the output and print the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 6. Generate output in dictionary format

# COMMAND ----------

               

# COMMAND ----------

# MAGIC %md
# MAGIC 7. Decode the output and print the result

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 8. Generate output in dictionary format with attentions

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####Update the model configs

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Check your current generation configs

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2. Update your model's generation configs

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####Where to find the complete list of generation configs
# MAGIC List of [Generation Configs](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig) in Hugging Face Transformer Documentation 

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>