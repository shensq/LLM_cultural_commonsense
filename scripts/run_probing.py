import argparse
import json
import re
import torch
import os
import openai

from api_keys import azure_key
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

def get_model(model_name):
    if re.match(r'meta-llama/Llama-2-\d+b-chat-hf', model_name):
        access_token = "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"
        tokenizer = AutoTokenizer.from_pretrained(f"{model_name}", use_auth_token=access_token)
        model = AutoModelForCausalLM.from_pretrained(f"{model_name}", torch_dtype=torch.float16, device_map="auto",use_auth_token=access_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
        model = AutoModelForCausalLM.from_pretrained(f"{model_name}", torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer
    


def load_data(task):
    if task == "association":
        file_path = "../data/genericskb/inputs_lang_country_association.json"
    elif task == "verification":    
        file_path = "../data/genericskb/inputs_lang_country_verification.json"
    elif task == "qa":
        file_path = "../data/geomlama/inputs_lang_country.json"
    elif task == "country_prediction":
        file_path = "../data/candle/inputs_lang_country.json"

    with open(file_path, 'r', encoding='utf-8') as json_file:
        inputs_lang_country = json.load(json_file)

    return inputs_lang_country

def save_data(results, model_name, task):
    # Just for saving data replace the / with - in the model name to avoid an extra folder 
    model_name = model_name.replace("/", "-")
    folder_path = f"../results/{task}/{model_name}"
    
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    file_name = f"{model_name}_{time_str}.json"
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    
    if os.path.exists(file_path):
        raise FileExistsError(f"The file {file_path} already exists.")

    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    return


def chat_completion(user_prompt, model_name, tokenizer, model):
    
    # Add the specific format for LLAMA-2
    if re.match(r'meta-llama/Llama-2-\d+b-chat-hf', model_name):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        # system_prompt = "You are a helpful, respectful and honest assistant. "
        system_prompt = ""
        prompt = B_INST + B_SYS + system_prompt +E_SYS + user_prompt + " " + E_INST
    elif re.match(r'lmsys/vicuna-\d+b-v1.5', model_name):
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        prompt = system_prompt + " " + "USER: " + user_prompt + " " + "ASSISTANT:"
    elif re.match(r'tiiuae/falcon-\d+b-instruct', model_name):
        system_prompt = ""
        prompt = system_prompt + "User: " + user_prompt + "\n\nAssistant:"
    
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    # Generate a response from the model
    output = model.generate(input_ids, max_new_tokens=72, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    # Decode and print the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt):]


def get_outputs(inputs_lang_country, model_name, tokenizer, model):
    results = {}
    for lang in inputs_lang_country:
        results[lang] = {}
        for country in inputs_lang_country[lang]:
            results[lang][country] = []
            print(lang, country)
            with tqdm(total=len(inputs_lang_country[lang][country][:500]), mininterval=30, maxinterval=60) as pbar:
                for sample in inputs_lang_country[lang][country][:500]:
                    messages = sample['prompt']
                    generated = chat_completion(messages, model_name, tokenizer, model)
                    results[lang][country].append({"prompt": sample['prompt'], "answer":sample['answer'], "generated":generated})
                    pbar.update(1)
                    
    return results


def get_completion_openai(messages, engine): 
    try:
        
        response = openai.ChatCompletion.create(
            engine=engine,
            messages = messages,
            temperature=1,
            max_tokens=128,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(e)
        return ""
    

def get_outputs_openai(inputs_lang_country, engine):
    results = {}
    for lang in inputs_lang_country:
        results[lang] = {}
        for country in inputs_lang_country[lang]:
            results[lang][country] = []
            print(lang, country)
            with tqdm(total=len(inputs_lang_country[lang][country][:500]), mininterval=30, maxinterval=60) as pbar:
                for sample in inputs_lang_country[lang][country][:500]:
                    messages = [{'role':'user', 'content':sample['prompt']},]
                    generated = get_completion_openai(messages, engine)
                    results[lang][country].append({"prompt": sample['prompt'], "answer":sample['answer'], "generated":generated})
                    pbar.update(1)
    return results


def setup_openai():
    # only ask for the api keys if using the azure api
    from api_keys import azure_key,api_base
    openai.api_type = "azure"
    openai.api_version = "2023-07-01-preview"
    openai.api_base = api_base
    openai.api_key = azure_key
    return


def main():
    parser = argparse.ArgumentParser(description='Select the model and the task to run.')

    parser.add_argument('-m', '--model_name', help='the model to run experiments on', default="")
    parser.add_argument('-t', '--task', help='the task to run experiments on', default="association")
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')

    args = parser.parse_args()
    print(args)
    data = load_data(args.task)
    
    if args.model_name in ["gpt-35-turbo", "gpt-4"]:
        setup_openai()
        results = get_outputs_openai(data, args.model_name)
    else:
        model, tokenizer = get_model(args.model_name)
        results = get_outputs(data, args.model_name, tokenizer, model)
        
    save_data(results, args.model_name, args.task)
    
    return
    




if __name__=="__main__":
    main()