import argparse
from datasets import load_dataset as load_dataset_hug_face
from torch.utils.data import random_split, DataLoader

from dataset.dataset_utils_lang_translation.bilingual_dataset import BilingualDataset

from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_all_sentences(ds, lang):
    '''

    Args:
        ds: It is raw dataset used form Hugging Face where data is stored in form ---> ds[idx]['translation'][language_name]
                                                                    ----> language name we can use en : english
                                                                    ---->                          it : italian
        lang : as mentioned above in ds parameter, necessary requirement for which parameters we are talking about

    Returns:

    '''
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(args : argparse.ArgumentParser, ds, lang : str) :

    '''

    Args:
        args:  contains all the argumnets needed in main file ---> For more reference refer to main.py
        ds: dataset are considering ---> ds will generally contain some format where each id from 1 src lang to 1 target lang
        lang: describe for which langauage we are building tokenizer ---> can be src or dest or can be en to it

    Returns: tokenizer where each word is represented with some id ----> e.g. ant ---> 1 the ---> 2 and so on ....

    '''

    tokenizer_path = Path(args.tokenizer_file.format(lang))

    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def load_dataset(args : argparse.ArgumentParser) :

    # Get the details for dataset
    ds_raw = load_dataset_hug_face(f"{args.datasource}", f"{args.lang_src}-{args.lang_tgt}", split='train')
    print(f"Total sentences  : {int(len(ds_raw))}")
    print("Printing sentence details ")
    idx = 5
    print(
        f"Data id : {ds_raw[idx]['id']} \n English : {ds_raw[idx]['translation']['en']} \n Italian : {ds_raw[idx]['translation']['it']} ")

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(args, ds_raw, args.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(args, ds_raw, args.lang_tgt)

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][args.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][args.lang_tgt]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')


    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, args.lang_src, args.lang_tgt,
                                args.seq_len)
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, args.lang_src, args.lang_tgt,
                              args.seq_len)

    train_dataloader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt