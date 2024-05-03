import lightning as L
from omegaconf import DictConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.optim import AdamW
import evaluate
import torch

class Seq2SeqModel(L.LightningModule):
    def __init__(self, model_config_params: DictConfig, tokenizer: AutoTokenizer, output_path: str):
        super(Seq2SeqModel, self).__init__()
        self.lr = model_config_params.lr
        self.tokenizer = tokenizer
        self.lang_code = model_config_params.lang_code
        self.output_path = output_path
        self.test_pairs = []
        if model_config_params.use_qlora:
            bnb_config = BitsAndBytesConfig(
                bnb_4bit_compute_dtype= torch.bfloat16,
                **model_config_params.bnb
            )
            
            config = LoraConfig(
                **model_config_params.lora
            )

            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_config_params.model_name,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                device_map={"": 0}
            )
            
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, config)

        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_config_params.model_name)
        self.model = model
        self.bleu_metric = evaluate.load('sacrebleu')
        self.rouge_metric = evaluate.load('rouge')

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)

        # Generate predictions
        generated_tokens = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang_code])
        decoded_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang_code])
        
        # Remove -100 indices (labels for padding tokens)
        decoded_labels = [label.replace(self.tokenizer.pad_token, '') for label in decoded_labels]
        decoded_labels_bleu = [[label] for label in decoded_labels]
        
        # Update metrics
        self.bleu_metric.add_batch(predictions=decoded_preds, references=decoded_labels_bleu)
        self.rouge_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def on_validation_epoch_end(self):
        bleu_result = self.bleu_metric.compute()
        rouge_result = self.rouge_metric.compute()

        # Log metrics
        self.log_dict({'val_bleu': bleu_result['score'], 'val_rouge': rouge_result['rougeL']}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Generate predictions
        generated_tokens = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang_code])
        #decoded_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.lang_code])
        #decoded_labels_bleu = [[label] for label in decoded_labels]
        
        # Remove -100 indices (labels for padding tokens)
        #decoded_labels = [label.replace(self.tokenizer.pad_token, '') for label in decoded_labels]

        test_pairs_batch = [pred_sentence.strip() for pred_sentence in decoded_preds]
        self.test_pairs.extend(test_pairs_batch)

    def on_test_epoch_end(self):
        print("Generating predictions")
        with open(self.output_path, 'w') as f:
            for test_pair in self.test_pairs:
                f.write(test_pair+'\n')
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min'),
            'name': 'learning_rate_log',
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]