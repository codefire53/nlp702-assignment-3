from omegaconf import DictConfig, OmegaConf
import hydra
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor
from tqdm import tqdm
import csv
import re
from tokenizers_dir.tokenizers import load_tokenizers
from models.models import Seq2SeqModel
from dataset.dataset import MadarDataset

def extract_language_code(text: str):
    pattern = re.compile(r'#(.*?)\.')
    matches = pattern.findall(text)
    return matches[0]


@hydra.main(version_base=None, config_path="./confs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(42)

    source_tokenizer_langs = [extract_language_code(filename) for filename in cfg.datasets.train_files]
    target_tokenizer_lang = cfg.datasets.target_lang
    test_source_tokenizer_langs = [extract_language_code(filename) for filename in cfg.datasets.test_files]
    source_tokenizers, target_tokenizer = load_tokenizers(cfg.models.model_name, source_tokenizer_langs, target_tokenizer_lang)
    test_source_tokenizers, _ = load_tokenizers(cfg.models.model_name, test_source_tokenizer_langs, target_tokenizer_lang)

    madar_dataset = MadarDataset(cfg.datasets, target_tokenizer, source_tokenizers, test_source_tokenizers)
    madar_dataset.load_dataloader(cfg.datasets.batch_size)
    train_dataloader, val_dataloader, test_dataloader = madar_dataset.train_dl, madar_dataset.val_dl, madar_dataset.test_dl
    model = Seq2SeqModel(cfg.models, target_tokenizer, cfg.output_path)

    # wandb logger for monitoring
    wandb_logger = WandbLogger(**cfg.loggers)
    
    # callbacks
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint)
    early_stop = EarlyStopping(**cfg.earlystopping)
    rich = RichProgressBar()
    lr_monitor = LearningRateMonitor(**cfg.lr_monitors)
    callbacks = [checkpoint_callback, early_stop, lr_monitor, rich]

    trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **cfg.trainer)

    if cfg.do_train:
        if hasattr(cfg, "checkpoint_file"):
            trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.checkpoint_file)
        else:
            trainer.fit(model, train_dataloader, val_dataloader)
    
    if cfg.do_test:
        trainer = Trainer(accelerator="gpu", devices=1)
        test_dataset = load_test_dataset(cfg.dataset)
        if cfg.do_train:
            # load best checkpoint
            model = Seq2SeqModel.load_from_checkpoint(checkpoint_callback.best_model_path, model_config_params=cfg.models, tokenizer=model.tokenizer, adapter_config_params=cfg.adapters, output_path=model.output_path)
        elif hasattr(cfg, "checkpoint_file"):
            model = Seq2SeqModel.load_from_checkpoint(cfg.checkpoint_file, model_config_params=cfg.models, tokenizer=model.tokenizer, adapter_config_params=cfg.adapters, output_path=model.output_path)
        trainer.test(model, test_dataloader)

if __name__ == "__main__":
    main()
