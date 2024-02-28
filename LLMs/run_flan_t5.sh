#!/bin/bash

#SBATCH --qos=default
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=muffin_2sccot_flan-t5-large4.stdout
#SBATCH --job-name=prompt_muffin_2sccot_flan-t5-large4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n102

python prompting_muffin.py \
  --model_name_or_path /XXX/flan-t5-large \
  --template_name 2S_CCOT \
  --token hf_XXXXXXXXX
