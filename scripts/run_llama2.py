import torch
import json 
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


access_token = "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"
model_name = "Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model_name}", use_auth_token=access_token)
model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_name}", torch_dtype=torch.float16, load_in_8bit=True, device_map="auto",use_auth_token=access_token)
# model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_name}", torch_dtype=torch.float16, device_map="auto",use_auth_token=access_token)

def chat_completion(user_prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    # system_prompt = "You are a helpful, respectful and honest assistant. "
    system_prompt = "Answer in the json format of {\"Answer\": }."
    # system_prompt = ""

    prompt = B_INST + B_SYS + system_prompt +E_SYS + user_prompt + E_INST

    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    # Generate a response from the model
    output = model.generate(input_ids, max_new_tokens=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    # Decode and print the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt):]

def main():

    def set_up_folders():
        folder_path = "../results"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        dataset = ["geomlama",'candle','genericskb']
        for d in dataset:
            if not os.path.exists(folder_path+'/'+d):
                os.makedirs(folder_path+'/'+d)
            if not os.path.exists(folder_path+'/'+d+'/'+model_name):
                os.makedirs(folder_path+'/'+d+'/'+model_name)


    set_up_folders()
    # run geomlama
    with open("../data/geomlama/inputs_lang_country.json", 'r', encoding='utf-8') as json_file:
        inputs_lang_country = json.load(json_file)


    lang_country_mapping = dict(zip(['zh', 'en', 'hi', 'fa', 'sw'], ['china', 'us', 'india', 'iran', 'kenya']))

    results = {}
    for lang in inputs_lang_country:
        results[lang] = {}
        for country in inputs_lang_country[lang]:
            if country == lang_country_mapping[lang] or lang=='en': # keep only the English ones and the native languages 
                results[lang][country] = []
                print(lang, country)
                for sample in tqdm(inputs_lang_country[lang][country][:]):
                    messages = sample['prompt']
                    generated = chat_completion(messages)
                    results[lang][country].append({"prompt": sample['prompt'], "answer":sample['answer'], "generated":generated})

    with open(f"../results/geomlama/{model_name}/llama2_results.json", 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)


    # run candle
    with open("../data/candle/inputs_lang_country.json", 'r', encoding='utf-8') as json_file:
        inputs_lang_country = json.load(json_file)
    results = {}
    for lang in inputs_lang_country:
        results[lang] = {}
        for country in inputs_lang_country[lang]:
            results[lang][country] = []
            print(lang, country)
            for sample in tqdm(inputs_lang_country[lang][country][:]):
                messages = sample['prompt']
                generated = chat_completion(messages)
                results[lang][country].append({"prompt": sample['prompt'], "answer":sample['answer'], "generated":generated})

    with open(f"../results/candle/{model_name}/llama2_results.json", 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)



    with open("../data/genericskb/inputs_lang_country_verification.json", 'r', encoding='utf-8') as json_file:
        inputs_lang_country = json.load(json_file)
    results = {}

    for lang in inputs_lang_country:
        results[lang] = {}
        for country in inputs_lang_country[lang]:
            results[lang][country] = []
            print(lang, country)
            for sample in tqdm(inputs_lang_country[lang][country][:500]):
                messages = sample['prompt']
                generated = chat_completion(messages)
                results[lang][country].append({"prompt": sample['prompt'], "answer":sample['answer'], "generated":generated})

    with open(f"../results/genericskb/{model_name}/llama2_verification_results.json", 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)


    with open("../data/genericskb/inputs_lang_country_association.json", 'r', encoding='utf-8') as json_file:
        inputs_lang_country = json.load(json_file)
    results = {}

    for lang in inputs_lang_country:
        results[lang] = {}
        for country in inputs_lang_country[lang]:
            results[lang][country] = []
            print(lang, country)
            for sample in tqdm(inputs_lang_country[lang][country][:500]):
                messages = sample['prompt']
                generated = chat_completion(messages)
                results[lang][country].append({"prompt": sample['prompt'], "answer":sample['answer'], "generated":generated})


    with open(f"../results/genericskb/{model_name}/llama2_association_results.json", 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

if __name__=="__main__":
    main()