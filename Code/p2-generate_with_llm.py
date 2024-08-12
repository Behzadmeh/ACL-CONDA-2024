import json
import re
import random
import time

from pprint import pprint

from transformers import AutoTokenizer  #, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch

from openai import OpenAI
openai_client = OpenAI(
    api_key="",
)

from constants import *


def assure_path_exists(path):
    """Assure that a path exists, eventually creating all the subdirs if needed.

    :param path:
    :return:
    """
    dir = os.path.expanduser(path)  # os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_llm_pipe(llm_key):
    model_id = LLM_K_TO_MODEL_ID[llm_key]
    # print(f"\tLLM: About to load {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    my_pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        )

    return my_pipeline, tokenizer


def generate_with_llama2(my_data, my_pipeline, tokenizer):
    llm_key = my_data[DATA_K_CFG][DATA_K_CFG_LLM],
    
    instances = my_data[DATA_K_INSTANCES]
    instances_sorted_items = sorted(instances.items())
    ###
    # Filter for using only few instances, subsetting by template ID
    instances_sorted_items = [tup for tup in instances_sorted_items if tup[0].startswith("instance_1") or tup[0].startswith("instance_9")]
    ###
    prompts = [query for _, (query,_) in instances_sorted_items]

    sequences = my_pipeline(
        prompts,
        do_sample=False,
        # top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        # https://discuss.huggingface.co/t/pipeline-max-length/74470
        max_length=MAX_LENGTH_GEN,
        # https://huggingface.co/docs/transformers/en/pad_truncation
        # https://huggingface.co/docs/transformers/main_classes/tokenizer
        # https://stackoverflow.com/questions/67849833/how-to-truncate-input-in-the-huggingface-pipeline
        # https://stackoverflow.com/questions/74018095/how-to-know-if-huggingfaces-pipeline-text-input-exceeds-512-tokens
        truncation=True,
    )

    instances_w_llm_gen = dict()
    for i, (instance_id, (query, true_result)) in enumerate(instances_sorted_items):
        seq = sequences[i][0]
        llm_gen = seq['generated_text']
        # Extend the data dictionary with the LLM generations:
        llm_result = llm_gen.index(str(true_result)) if str(true_result) in llm_gen else -1  # CHANGE for some other postprocessing of llm_gen with regexp, etc. to obtain the actual number
        instances_w_llm_gen[instance_id] = (query, true_result, llm_gen, llm_result)

    my_data[DATA_K_INSTANCES] = instances_w_llm_gen
    #pprint(instances_w_llm_gen)


# ##### GPT

# https://github.com/openai/openai-python
# https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion.py
def get_gpt_completion(prompt, model_id, temperature=0.0, wait=1, return_used_tokens=False):
    time.sleep(wait)
    messages = [{"role": "user", "content": prompt}]

    completion = openai_client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
    )
    content = completion.choices[0].message.content

    if return_used_tokens:
        return content, completion.usage.prompt_tokens, completion.usage.completion_tokens
    else:
        return content, _ , _


def generate_with_gpt(my_data, get_cost=False):
    llm_key = my_data[DATA_K_CFG][DATA_K_CFG_LLM]
    assert llm_key in CFG_LLMS_GROUP_GPT
    model_id = LLM_K_TO_MODEL_ID[llm_key]

    instances = my_data[DATA_K_INSTANCES]
    instances_sorted_items = sorted(instances.items())
    ###
    # Filter for using only few instances, subsetting by template ID
    #instances_sorted_items = [tup for tup in instances_sorted_items if tup[0].startswith("instance_0") or tup[0].startswith("instance_1")][990:1010]

    instances_w_llm_gen = dict()
    cost_per_q_in_this_config = dict()
    for i, (instance_id, (query, true_result)) in enumerate(instances_sorted_items):
        prompt = query
        gpt_result, n_prompt_tokens, n_completion_tokens = get_gpt_completion(prompt,
                                                                          model_id,
                                                                          return_used_tokens=True)

        print("-"*5)
        print(f"\tEvaluating {i+1}-th/{len(instances_sorted_items)} instance, ({instance_id}): '{query}?'")
        print(f"\t\tLLM generated: '{gpt_result}'")
        if get_cost:
            cost_for_this_q = (n_prompt_tokens * LLMS_GPT_COST_PER_TOKEN_IN[llm_key] + 
                               n_completion_tokens * LLMS_GPT_COST_PER_TOKEN_OUT[llm_key])
            print(f"\t\tCost for this question = ${cost_for_this_q}")
            cost_per_q_in_this_config[instance_id] = cost_for_this_q

        llm_gen = gpt_result
        # Extend the data dictionary with the LLM generations:
        llm_result = llm_gen.index(str(true_result)) if str(true_result) in llm_gen else -1  # CHANGE for some other postprocessing of llm_gen with regexp, etc. to obtain the actual number
        instances_w_llm_gen[instance_id] = (query, true_result, llm_gen, llm_result)
        print('{'+f"{instance_id}: {instances_w_llm_gen[instance_id]}"+'}')
        print("\n")

    my_data[DATA_K_INSTANCES] = instances_w_llm_gen

    return cost_per_q_in_this_config



def main(llm_keys):
    assure_path_exists(DIR_EXPERS)
    assure_path_exists(DIR_LLM_GEN_FOR_INSTANCES)

    for llm_key in llm_keys:
        print(f"*** LLM = {llm_key}")
        my_data = json.load(open(FPATH_DATA_INSTANCES))
        # Add a portable part about configuration/parameters
        # Now it's very small but it could have more params later
        my_data[DATA_K_CFG] = {
            DATA_K_CFG_LLM: llm_key
        }
        try:
            if llm_key in CFG_LLMS_GROUP_LLAMA2:
                my_pipeline, tokenizer = load_llm_pipe(llm_key)
                generate_with_llama2(my_data, my_pipeline, tokenizer) 
            elif llm_key in CFG_LLMS_GROUP_GPT:
                cost_per_q_in_this_config = generate_with_gpt(my_data, get_cost=True)
                cost = sum(cost_per_q_in_this_config.values())
                print(f"Total costs for this config = ", cost)
            # Dump, now after full RAG
            fpath_out = FPATH_RES_LLM_GEN.substitute(
                llm=my_data[DATA_K_CFG][DATA_K_CFG_LLM],
                mnt=MAX_NEW_TOKENS,
            )
            json.dump(my_data, open(fpath_out, 'w'), indent=4)
        except Exception as error:
            print("\t! An exception occurred:", type(error).__name__)
            print(error)

        if llm_key in CFG_LLMS_GROUP_LLAMA2:
            # https://discuss.huggingface.co/t/clear-gpu-memory-of-transformers-pipeline/18310
            # https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/discussions/25
            del my_pipeline
            torch.cuda.empty_cache()


if __name__ == '__main__':
    CFG_LLMS = [#CFG_LLM_LLAMA2_7B_CHAT,
                #CFG_LLM_LLAMA2_13B_CHAT,
                #CFG_LLM_TINY_LLAMA_CHAT,
                #CFG_LLM_GPT_3_5,
                #CFG_LLM_GPT_4,
                ]
    main(CFG_LLMS)

