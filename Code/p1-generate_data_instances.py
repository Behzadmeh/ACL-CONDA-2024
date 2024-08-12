import random
import json

def generate_number(length): 
        return int("".join([str(random.randint(1,9))]  # first digit can't be zero
                   +[str(random.randint(0,9)) for _ in range(length-1)]))
    
data = dict()

# ### Define the templates
# Shape of data["1-Rephrasing_templates"] dict:
# {<template ID>: (<string template>, <template difficulty when produced by Behzad with GPT-4>)}
data["1-Rephrasing_templates"] = { 
    0: ("[term_1]+[term_2]=", "Basic"), 
    1: ("Can you please add [term_1] and [term_2] together?", "Easy"), 
    2: ("Please combine [term_1] with [term_2].", "Easy"), 
    3: ("Find the sum of [term_1] and [term_2].", "Easy"), 
    4: ("Add up two numbers: [term_1] and [term_2].", "Medium"), 
    5: ("Combine these two numbers: the first is [term_1], the second is [term_2].", "Medium"), 
    6: ("Please work out the total of [term_1] and [term_2].", "Medium"), 
    7: ("Please determine the numeric sum of [term_1] and [term_2].", "Difficult"), 
    8: ("Proceed to identify the aggregated total of the numbers [term_1] and [term_2].", "Difficult"), 
    9: ("Perform an addition operation on the numerical values [term_1] and [term_2].", "Difficult"), 
}

# ### Create all the terms per each desired length and compute their sum
# Shape of data["2-Term_pairs_and_sum"] dict:
# {<pair ID>: ((<term 1>, <term 2>), <term 1> + <term 2>)}
LOWER_LEN = 5
UPPER_LEN = 14
lengths = [x for x in range(LOWER_LEN, UPPER_LEN+1)]
INSTANCES_PER_LENGTH = 100
pairs_and_sum = dict()
i = 0
for length in lengths: 
    for i in range(0, INSTANCES_PER_LENGTH):
        rdm_length_1 = random.randint(1,length - 1)
        rdm_length_2 = length - rdm_length_1
        n_i_1 = generate_number(rdm_length_1)
        n_i_2 = generate_number(rdm_length_2)
        assert(len(str(n_i_1)) + len(str(n_i_2)) == length)
        pairs_and_sum[f"{length}_{i+1}"] = (((n_i_1, n_i_2), n_i_1+n_i_2))

data["2-Term_pairs_and_sum"] = pairs_and_sum

# ### Create all the actual input-output data instances, by instantiating each template with each pair of terms
# Shape of data["3-Instances"] dict:
# {<instance ID>: (<instance input, i.e. the string to prompt an LLM>, <instance output, i.e. the sum result>)},
# where the format of <instance ID> is "<template ID>_<pair's combined length>_<pair ID>"
instances = dict()
for tpt_id, tpt in sorted(data["1-Rephrasing_templates"].items()): 
    for (n_id, ((term_1, term_2), sum_result)) in sorted(data["2-Term_pairs_and_sum"].items()): 
        instance_id = f"instance_{tpt_id}_{n_id}" 
        instances[instance_id] = (tpt[0].replace("[term_1]", str(term_1)).replace("[term_2]", str(term_2)), sum_result) 

data["3-Instances"] = instances

# Ok. Dump
json.dump(data, open("../data/p1-templates+numbers+instances.json", 'w'), indent=4)
# (!) NOTE: when loading the JSON file, dict keys that were integers will be loaded as strings
