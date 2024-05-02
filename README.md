# Setup
Setup conda environment
```
conda create -n aradialect-mt python=3.10
```
Install dependencies
```
pip install -r requirements.txt
```

# How to run experiment
If you want to run the m2m100 model, you can adjust the `m2m100_madar.yaml` then run
```
python main.py --config-name m2m100_madar
```
If you want to run the nllb model, you can adjust the `nllb_madar.yaml` then run
```
python main.py --config-name nllb_madar
```

# Evaluation 
To evaluate your prediction you can just run like this
```
cd dataset
python eval_score.py ./val/NADI2024_subtask3_dev2_GOLD.txt ./predictions/nllb_madar_dev2.txt ./val/UBC_subtask3_dev_2.txt
```