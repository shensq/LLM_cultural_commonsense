import os
import re
import json 
import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm 


def get_base_samples():
    def separate_by_country(df):
        outputs = {}
        country_list = ["us",'china','india','iran','kenya']
        for i, country in enumerate(country_list):
            outputs[country] = df.iloc[i::5].values.tolist()
        return outputs
    # Get the base samples from **_lang.tsv 
    files = glob('../data/geomlama/gd_prompt_??.tsv')
    samples = {}
    for file in files:
        lang = file.split('_')[-1].split('.')[0]
        df = pd.read_csv(file, sep='\t', header=0)
        df = df[['Prompt','Ans', 'Candidate Ans']]
        df.dropna(inplace=True)
        df = df.reset_index(drop=True)
        samples[lang] = df
        samples[lang] = separate_by_country(samples[lang])
    return samples


def get_aug_samples():
    def separate_by_country(l):
        outputs = {}
        country_list = ["us",'china','india','iran','kenya']
        for i, country in enumerate(country_list):
            outputs[country] = l[i::5]
        return outputs
    # Get the masked input samples from **_lang_aug.tsv 
    files = glob('../data/geomlama/gd_prompt_??_aug.tsv')
    samples = {}
    for file in files:
        lang = file.split('_')[-2].split('.')[0]
        data = open(file, 'r').readlines()
        samples[lang] = data
        samples[lang] = separate_by_country(samples[lang])

    return samples


def main():
    samples_lang_country =  get_base_samples()

    aug_samples = get_aug_samples()

    samples_merged = {}

    for lang in samples_lang_country:
        samples_merged[lang] = {}
        for country in samples_lang_country[lang]:
            samples_merged[lang][country] = []
            # print(lang, country, len(samples_lang_country[lang][country]))
            
            for i, sample in enumerate(samples_lang_country[lang][country]):
                question, answer, options = sample
                samples_merged[lang][country].append(sample)
                for aug_sample in aug_samples[lang][country][i*4:i*4+4]:
                    samples_merged[lang][country].append([aug_sample, answer, options])

    # store the merged samples to json file
    with open("../data/geomlama/merged_lang_country.json", 'w', encoding='utf-8') as json_file:
        json.dump(samples_merged, json_file, ensure_ascii=False, indent=4)

    
    prompt1 = {'zh':"选择最佳选项以填充以下句子中的空白：",
           'en':"Choose the best option to fill in the blank in the following sentence: ",
           'hi':"निम्नलिखित वाक्य में रिक्त स्थान को भरने के लिए सबसे अच्छा विकल्प चुनें: ",
           'fa':"بهترین گزینه را برای پر کردن جای خالی در جمله زیر انتخاب کنید: ",
           'sw':"Chagua chaguo bora la kujaza tupu katika sentensi ifuatayo: ",
           }


    prompt2 = {'zh':"选项：",
                'en':"Options: ",
                'hi':"विकल्प: ",
                'fa':"گزینه: ",
                'sw':"Machaguo: ",
                }


    prompt3 = {'zh':"以 {“答案”： } 的 json 格式回答。",
                'en':"Answer in the json format of {\"Answer\": }.",
                'hi':"{\"उत्तर\": } के json स्वरूप में उत्तर दें।",
                'fa': "پاسخ در قالب json {\"پاسخ\": }.",
                'sw':"Jibu katika umbizo la json la {\"Answer\": }.",
            }

    inputs_lang_country = {}
    for lang in samples_merged:
        inputs_lang_country[lang] = {}
        for country in samples_merged[lang]:
            inputs_lang_country[lang][country] = []
            for sample in samples_merged[lang][country]:
                inputs_lang_country[lang][country].append({'prompt':prompt1[lang] + sample[0] + '\n' +prompt2[lang] + sample[2] +'\n' +  prompt3[lang], 'answer':sample[1]})

    with open("../data/geomlama/inputs_lang_country.json", 'w', encoding='utf-8') as json_file:
        json.dump(inputs_lang_country, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()