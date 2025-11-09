# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://learningjournal.github.io/pub-resources/logos/scholarnest_academy.jpg" alt="ScholarNest Academy" style="width: 1400px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Prepare for text generation

# COMMAND ----------

pip install --upgrade transformers

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from transformers import pipeline, set_seed
pipe = pipeline("text-generation", model = 'meta-llama/Llama-3.2-1B-Instruct',
                token="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")                
set_seed(45)           

# COMMAND ----------

# MAGIC %md
# MAGIC ###1. Greedy Search
# MAGIC Greedy search is the simplest decoding method. It selects the word with the highest probability as its next word. However, this approach could miss the high probability word hidden behind the current low probability word. This approach could be useful in translation and speech recognition. However, open ended text generation may suffer with repetitive sequences.
# MAGIC
# MAGIC #####This type of search can be achieved using num_beams=1 and do_sample=False
# MAGIC
# MAGIC ![](/files/images/greedy_search.jpg)

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Check your model's generation config

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 2. Write a story using the greedy search. Prompt is already provided.

# COMMAND ----------

prompt="""
Write a short story using the given clue. Do not generate incomplete sentances instead stop early.
Clue: I enjoy walking with my cute dog
Story:
"""


# COMMAND ----------

# MAGIC %md
# MAGIC ###2. Beam Search
# MAGIC Beam search reduces the risk of missing hidden high probability word sequences by tracking num_beams at each step and eventually choosing the hypothesis that has the overall highest probability. Beam search produce better results when the lenght of the result is predictable such as summarization task.
# MAGIC
# MAGIC #####This type of search can be achieved using num_beams>1 and do_sample=False
# MAGIC
# MAGIC ![](/files/images/beam_search.jpg)

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Write a story using the beam search. Prompt is already provided.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ###3. Sampling
# MAGIC Sampling introduces a randomness in choosing the next token. It will randomly pick the next word according to its conditional probability distribution instead of choosing the highest probility. Sampling is a good approach for introducing creativity or a surprize element in the output.
# MAGIC
# MAGIC #####This type of search can be achieved using do_sample=True

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Write a story using the sampling search. Prompt is already provided.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####3.1 Temprature
# MAGIC The temperature is a parameter that controls the randomness and creativity of the model's output. You can lower the element of randomeness by reducing the temprature.

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC 3.1. Adjust temprature to reduce the element of randomeness and get more realistic.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2 Top_p
# MAGIC
# MAGIC Top-p sampling chooses from the smallest possible set of words whose cumulative probability exceeds the given probability p. Reducing the top_p will explore less options, become more deterministic and factual but behaves like reducing the number of beams in the beam search. 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ####Where can I find a list of available decoding strategies?
# MAGIC List of [Decoding Strategies](https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies) in Hugging Face Transformer Documentation 

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2021-2025 <a href="https://www.scholarnest.in/">ScholarNest</a>. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation.</a><br/>
# MAGIC Databricks, Databricks Cloud and the Databricks logo are trademarks of the <a href="https://www.databricks.com/">Databricks Inc.</a><br/>
# MAGIC Hugging Face, Hugging Face Logo, Hugging Face Hub are trademarks of the <a href="https://huggingface.co/"> Hugging Face Inc. </a>
# MAGIC <br/>
# MAGIC <a href="https://www.scholarnest.in/pages/privacy">Privacy Policy</a> | <a href="https://www.scholarnest.in/pages/terms">Terms of Use</a> | <a href="https://www.scholarnest.in/pages/contact">Contact Us</a>