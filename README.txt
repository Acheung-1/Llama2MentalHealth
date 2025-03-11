
Andrew Cheung and Shi Bin Lim

Our project trains will fine-tune the opensourced Llama2 model

We will train the models with the following 3 datasets:

1) (PHR) phr_mental_therapy_dataset: https://huggingface.co/datasets/vibhorag101/phr_mental_therapy_dataset
2) (COUNSELING) mental_health_counseling_conversation: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations?row=0
3) (REDDIT) Data scraped from subreddits (r/offmychest, r/advice, r/mentalhealth, r/confessions, r/self): web_scraper.py obtains this data

We will TRAIN 3 models on a mixture of data as follows: 
1) COUNSELING (first 1000 rows)
2) COUNSELING (first 200 rows) and PHR (first 1000 rows)
3) COUNSELING (first 200 rows) and REDDIT (1000)

And EVALUATE them together with a Vanilla model using BLEU score on UNSEEN counseling data (last 500 rows of COUNSELING dataset) that has professional responses.


4 NLP CLASS CONCEPTS:
[1. Syntax] Train and test input will be tokenized and fed to the model.
[2. Semantics] Instruction-fine tuning the models gives instruction on how the language model should respond. 
[3. Transformer LLMs] Llama2 (LLMs) 
[4. Application] Evaluator method by BLEU score to identify which model produces an output most similar to a psychologist response.


The code we have provided below trains the models, evaluates BLEU score of their response, and runs on Google Co-lab.

LLAMA2:
llama_2_vanilla.py
finetuned_llama_2_v1.py
finetuned_llama_2_v2.py
finetuned_llama_2_v3.py

EXTRA:
web_scraper.py (Gets data from reddit API)
BLEUScore.py
