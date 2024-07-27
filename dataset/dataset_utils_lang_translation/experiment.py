import argparse
from pathlib import Path

from dataset.dataset_utils_lang_translation.text_corpus_2 import train_model


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
}

parser = argparse.ArgumentParser(description="Describe dataset details")
args = parser.parse_args()

args.batch_size = 8
args.num_epochs = 20
args.lr = 10**-4
args.seq_len = 350
args.d_model = 512
args.datasource = 'opus_books'
args.lang_src = 'en'
args.lang_tgt = 'it'
args.model_folder = 'weights'
args.model_basename = 'tmodel_'
args.preload = 'latest'
args.tokenizer_file = 'tokenizer_{0}.json'
args.experiment_name = "/raid/home/namanmalpani/final_yr/ML_Project/runs/lang_translation-debug"


# Make sure the weights folder exists


train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)