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

For AraT5, we adjusted the notebook from [here](https://github.com/UBC-NLP/araT5/blob/main/examples/Fine_tuning_AraT5.ipynb). To train the model, run the following command:
```
python AraT5/run_trainier_seq2seq_huggingface.py \
        --learning_rate 2e-5 \
        --max_target_length 128 --max_source_length 128 \
        --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
        --model_name_or_path "UBC-NLP/AraT5v2-base-1024" \
        --output_dir "AraT5/FT" --overwrite_output_dir \
        --num_train_epochs 22 \
        --train_file "dataset/madar_train.tsv" \
        --validation_file "dataset/madar_dev.tsv" \
        --task "translate_into_MSA" --text_column "source" --summary_column "target" \
        --load_best_model_at_end --metric_for_best_model "eval_bleu" --greater_is_better True --evaluation_strategy steps --logging_strategy steps --predict_with_generate \
        --do_train --do_eval > log3.log 2>&1
```

To run inference on dev set, run the following command:
```
python AraT5/inference.py
```


# Evaluation 
To evaluate your prediction you can just run like this
```
cd dataset
python eval_score.py ./val/NADI2024_subtask3_dev2_GOLD.txt ./predictions/araT5_dev.txt ./val/UBC_subtask3_dev_2.txt
```

# Assets
To download our model checkpoints and predictions, you can download them [here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/mahardika_ihsani_mbzuai_ac_ae/EjqbeoMa5gRFlo8fI0tr_NMBstp4LKMYBtHF2jInfWgY4A?e=79RVeb).
