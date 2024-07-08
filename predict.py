import transformers
from transformers import logging as transformers_logging
import torch
import pandas as pd
import re

transformers_logging.set_verbosity_error()

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device='cuda'
)

def generate_prompt(review):
    prompt = [
        {
            "role": "assistant",
            "content": "You are an expert and have very rich experience in evaluating online hotel reviews, especially in its helpfulness, subjectivity, and coherence. Subjectivity quantifies the amount of personal opinion and factual information contained in the review text. Coherence refers to degree of effective grouping and arrangement of ideas in a logical order. Helpfulness indicates to what extent the information of the review text can help consumers make purchase decisions. For a given hotel review, please provide the integer scores of its helpfulness, subjectivity, and coherence respectively in range 1 (least degree) - 7 (largest degree), and please only provide these scores in order in this format: score of helpfulness, score of subjectivity, score of coherence, without any form of explanation or original review text. Here is an example answer for a review: 2,4,1"
        },
        {
            "role": "user",
            "content": "Here is a piece of hotel review:" + review
        }
    ]
    return prompt

def tokenize_review(review):
    prompt = generate_prompt(review)

    review_token = pipeline.tokenizer.apply_chat_template(
        prompt, 
        tokenize=False, 
        add_generation_prompt=True,
    )

    return review_token

def process_data():
    df = pd.read_excel('review_rule_data_for_llm_calculation.xlsx', nrows=500)
    reviews = df['content']
    messages = reviews.apply(tokenize_review).to_list()
    return messages

def extract_result(str):
    # Find all numeric characters in the string
    all_numbers = re.findall(r'\d', str)
    # Get the last 3 numeric characters
    result = all_numbers[-3:]
    return result

def predict(messages):
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id
    outputs = pipeline(
        messages,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=1.,
        top_p=0.9,
        batch_size=100
    )

    results = list(map(lambda x: extract_result(x[0]['generated_text']), outputs))

    return results

messages = process_data()
print(predict(messages[:10]))