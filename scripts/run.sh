#!/bin/bash
# choose from lmsys/vicuna-7/13b-v1.5 tiiuae/falcon-7/40b-instruct meta-llama/Llama-2-7/13b-chat-hf
model=$1
python3 run_probing.py --model_name $model --task association
python3 run_probing.py --model_name $model --task verification
python3 run_probing.py --model_name $model --task qa
python3 run_probing.py --model_name $model --task country_prediction