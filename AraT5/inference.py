import json
from datasets import load_dataset
import os
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import tqdm

def main():
    # evaluate the best model on test
    best_model = "checkpoint-15000"
    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained(best_model)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/AraT5-base")
    # read df without index
    test_df = pd.read_csv("../dataset/val/NADI2024_subtask3_dev.tsv", sep="\t")
    dialect_sentences = test_df["source_da"].tolist()
    # predict with the model with batch size 8
    msa_sentences = []
    for i in range(0, len(dialect_sentences), 8):
        batch = dialect_sentences[i : i + 8]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model.generate(**inputs)
        msa_sentences.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # save the predictions to txt file
    with open("../dataset/predictions/araT5_dev_new.txt", "w") as f:
        for sentence in msa_sentences:
            f.write(sentence + "\n")


if __name__ == "__main__":
    main()
