from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import datasets
import torch

class MadarDataset:
    def _tokenize_function(self, source_tokenizer, target_tokenizer, examples: list):
        model_inputs = source_tokenizer(examples['source'], add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
        if 'target' in examples:
            with target_tokenizer.as_target_tokenizer():
                labels = target_tokenizer(examples['target'], add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
                model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _load_dataset_from_files(self, filepaths, source_tokenizers, target_tokenizer, is_test):
        final_dataset = dict()
        delimiter = '\t'
        for idx, filepath in enumerate(filepaths):
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.read().splitlines()
            if not is_test:
                dataset_dict = {
                    'source':[],
                    'target':[]
                }
                for line in lines:
                    texts = line.split(delimiter)
                    dataset_dict['source'].append(texts[0].strip())
                    dataset_dict['target'].append(texts[1].strip())
            else:
                dataset_dict = {
                    'source':[]
                }
                for line in lines:
                    dataset_dict['source'].append(line.strip())
            dataset_subset = datasets.Dataset.from_dict(dataset_dict)
            tokenized_dataset_subset = dataset_subset.map(lambda examples: self._tokenize_function(source_tokenizers[idx], target_tokenizer, examples), batched=True, remove_columns=dataset_subset.column_names)
            for example in tokenized_dataset_subset:

                for col in tokenized_dataset_subset.features.keys():
                    if col not in list(final_dataset.keys()):
                        final_dataset[col] = []
                    final_dataset[col].append(example[col])
        return datasets.Dataset.from_dict(final_dataset)

    def __init__(self, dataset_config_params, target_tokenizer, source_tokenizers, test_source_tokenizers):
        self.train_dataset = self._load_dataset_from_files(dataset_config_params.train_files, source_tokenizers, target_tokenizer, False)
        self.val_dataset = self._load_dataset_from_files(dataset_config_params.val_files, source_tokenizers, target_tokenizer, False)
        self.test_dataset = self._load_dataset_from_files(dataset_config_params.test_files, test_source_tokenizers, target_tokenizer, True)
        self.tokenizer = target_tokenizer

    def _initialize_collator_module(self):
        return DataCollatorForSeq2Seq(self.tokenizer, padding='max_length', max_length=512, return_tensors='pt')

    def load_dataloader(self, batch_size):
        collator_fn = self._initialize_collator_module()
        # create dataloader
        self.train_dl = DataLoader(self.train_dataset, shuffle=True, collate_fn=collator_fn, batch_size=batch_size)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, collate_fn=collator_fn, batch_size=batch_size)
        self.test_dl = DataLoader(self.test_dataset, shuffle=False, collate_fn=collator_fn, batch_size=batch_size)
 