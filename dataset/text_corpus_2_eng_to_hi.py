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

    args.tokenizer_file = "/data/home/karmpatel/karm_8T/naman/demo/data/tokenizer/iitb_en2hi_tokenizer_{0}.json"

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

    # Re - initializing the dataset for english to hind dataset
    args.datasource = "cfilt"
    args.lang_src_name = "english"
    args.lang_tgt_name = "hindi"
    args.lang_src = "en"
    args.lang_tgt = "hi"

    # Get the details for dataset
    ds_raw_train = load_dataset_hug_face("cfilt/iitb-english-hindi", split='train')
    ds_raw_val = load_dataset_hug_face("cfilt/iitb-english-hindi", split='validation')
    ds_raw_test = load_dataset_hug_face("cfilt/iitb-english-hindi",split='test')

    print(f"Total Train sentences  : {int(len(ds_raw_train))}")
    print("Printing sentence details ")
    idx = 5
    print(
        f"Data id : {ds_raw_train[idx]['id']} \n English : {ds_raw_train[idx]['translation'][args.lang_src]} \n Hindi : {ds_raw_train[idx]['translation'][args.lang_tgt]} ")

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(args, ds_raw_train, args.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(args, ds_raw_train, args.lang_tgt)

    # Find the maximum length of each sentence in the source and target sentence
    max_len_train_src = 0
    max_len_train_tgt = 0
    max_len_val_src = 0
    max_len_val_tgt = 0

    for item in ds_raw_train:
        src_ids = tokenizer_src.encode(item['translation'][args.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][args.lang_tgt]).ids
        max_len_train_src = max(max_len_train_src, len(src_ids))
        max_len_train_tgt = max(max_len_train_tgt, len(tgt_ids))

    print(f'Max length of train source sentence: {max_len_train_src}')
    print(f'Max length of train target sentence: {max_len_train_tgt}')

    for item in ds_raw_val :
        src_ids = tokenizer_src.encode(item['translation'][args.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][args.lang_tgt]).ids
        max_len_val_src = max(max_len_val_src, len(src_ids))
        max_len_val_tgt = max(max_len_val_tgt, len(tgt_ids))

    print(f'Max length of val source sentence: {max_len_val_src}')
    print(f'Max length of val target sentence: {max_len_val_tgt}')

    args.seq_len = max(max_len_train_src, max_len_train_tgt, max_len_val_src, max_len_val_tgt) + 5


    # Keep 90% for training, 10% for validation
    train_ds_size = len(ds_raw_train)
    val_ds_size = len(ds_raw_val)

    train_ds = BilingualDataset(ds_raw_train, tokenizer_src, tokenizer_tgt, args.lang_src, args.lang_tgt, args.seq_len)
    val_ds = BilingualDataset(ds_raw_test, tokenizer_src, tokenizer_tgt, args.lang_src, args.lang_tgt, args.seq_len)

    train_dataloader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt