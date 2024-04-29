#!/bin/bash

#SBATCH --job-name=nllb_madar_nlp702-assignment3# Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l2-001
python main.py --config-name nllb_madar
