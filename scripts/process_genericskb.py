import json




def main():
    
    # specify the corresponding langauge
    country_lang_mapping = {"the United States":"en",
                    "India":'hi',
                    "China":'zh-Hans',
                    "Iran":'fa',
                    "Kenya":'sw',}
    # prompt for association, translated to 5 languages
    translated_prompt = {'zh-Hans': '伊朗、中国、美国、印度、肯尼亚等哪些国家最有可能说这种说法？仅选择一个国家/地区。以 {“Answer”： } 的 JSON 格式回答',
        'hi': 'किन देशों में यह कथन सत्य होने की सबसे अधिक संभावना है, ईरान, चीन, संयुक्त राज्य अमेरिका, भारत, केन्या? केवल एक देश का चयन करें। {"उत्तर": } के JSON प्रारूप में उत्तर दें',
        'fa': 'در کدام کشورها این بیانیه به احتمال زیاد درست است، ایران، چین، ایالات متحده، هند، کنیا؟ فقط یک کشور را انتخاب کنید. پاسخ در فرمت JSON {"پاسخ": }',
        'sw': 'Ni katika nchi gani kauli hii inaweza kuwa kweli, Iran, China, Marekani, India, Kenya? Chagua nchi moja tu. Jibu katika umbizo la JSON la {"Answer": }',
        'en': 'In which countries is this statement most likely to be true,  Iran, China, the United States, India, Kenya? Select only one country. Answer in the JSON format of {"Answer": }'}
    
    # an example will be like: '{assertion}' prompt
    # The file is stored in ../data/genericskb/inputs_lang_country_base.json, the translation code is omiited here.
    with open("../data/genericskb/inputs_lang_country_base.json", 'r', encoding='utf-8') as json_file:
        inputs_lang_country_base = json.load(json_file)

    
    inputs_lang_country = {}
    inputs_lang_country['en'] = {}


    # Add the prompt to all countries in native language
    for country,lang in country_lang_mapping.items():
        inputs_lang_country[lang] = {}
        inputs_lang_country[lang][country] = []
        for assertion in inputs_lang_country_base[lang][country]:
            prompt = assertion['prompt'] + ' ' +translated_prompt[lang]
            inputs_lang_country[lang][country].append({'prompt': prompt, 'answer': ""})

    with open("../data/genericskb/inputs_lang_country_association.json", 'w', encoding='utf-8') as json_file:
        json.dump(inputs_lang_country, json_file, ensure_ascii=False, indent=4)

    
    

if __name__=="__main__":
    main()
    

