import argparse
import pandas as pd
import re
import unicodedata
import string
import contractions
from datasets import Dataset
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import random_split, DataLoader
from tokenizers.pre_tokenizers import Whitespace

from dataset.dataset_utils_lang_translation.bilingual_dataset import BilingualDataset
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

from dataset.dataset_utils_lang_translation.bilingual_dataset2 import BilingualDataset2


# Preprocessing data

# Removing HTML Tags
def remove_html(text):
    if isinstance(text, str):

        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)
    else:
        return text

# Remove URL Tags
def remove_url(text):
    if isinstance(text,str):
        pattern = re.compile(r'https?://\S+|www\.\S+')
        return pattern.sub(r'',text)
    else:
        return

# Removing Non-Hindi and Alphanumeric Characters
def preprocess_text(text, language='english'):
    if not isinstance(text, str):
        return text

    if language == 'english':
        pattern = re.compile(r'[^a-zA-Z0-9\s]')
        return pattern.sub(r'', text)
    elif language == 'hindi':
        pattern = re.compile(r'[^\u0900-\u097F\s]')
        return pattern.sub(r'', text)
    else:
        raise ValueError("Unsupported Language, Supported languages are 'english' and 'hindi'")

def get_hindi_punctuations():
    hindi_punctuations = []
    for i in range(0x2000, 0x206f + 1):
        char = chr(i)
        if unicodedata.category(char) == 'Po':
            hindi_punctuations.append(char)
    return ''.join(hindi_punctuations)

def remove_punctuation(text, language='English'):
    if language == 'English':
        exclude_english = set(string.punctuation)
        return ''.join(char for char in text if char not in exclude_english)
    elif language == 'Hindi':
        hindi_punctuation = get_hindi_punctuations()
        return ''.join(char for char in text if char not in hindi_punctuation)

    else:
        raise ValueError("Unsupported Language, Supported languages are 'english' and 'hindi'")

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

def do_tokenization(text):
    token_words = word_tokenize(text)
    return ' '.join(token_words)

def remove_stopwords(text,language = 'english'):

    stop_words_english = set(stopwords.words('english'))
    stop_words_hindi = set(nltk.corpus.indian.words('hindi.pos'))

    if language == 'english':
        filtered_words_english = [word for word in text if word.lower() not in stop_words_english]
        return ' '.join(filtered_words_english)
    elif language == 'hindi':
        filterd_words_hindi = [word for word in text if word not in stop_words_hindi]
        return ' '.join(filterd_words_hindi)
    else:
        return ValueError("Unsupported Language, Supported languages are 'english' and 'hindi'")

def do_stemming(token_words):
    ps = PorterStemmer()
    words = token_words.split()
    return ' '.join(words)

def preprocess_data(file_name) :

    ds_raw = pd.read_csv(file_name)

    # Printing 5 samples
    print(ds_raw.sample(5))

    # Removing rows with null values
    ds_raw.dropna(inplace=True)
    # Lowercasing english
    ds_raw['English'] = ds_raw['English'].str.lower()
    # Removing HTML Tags
    ds_raw['English'] = ds_raw['English'].apply(remove_html)
    ds_raw['Hindi'] = ds_raw['Hindi'].apply(remove_html)
    # Remove URL Tags
    ds_raw['English'] = ds_raw['English'].apply(remove_url)
    ds_raw['Hindi'] = ds_raw['Hindi'].apply(remove_url)
    # Removing Non-Hindi and Alphanumeric Characters
    ds_raw['English'] = ds_raw['English'].apply(lambda x: preprocess_text(x, language='english'))
    ds_raw['Hindi'] = ds_raw['Hindi'].apply(lambda x: preprocess_text(x, language='hindi'))
    # Removing punctuations
    ds_raw['English'] = ds_raw['English'].apply(lambda x: remove_punctuation(x, language='English'))
    ds_raw['Hindi'] = ds_raw['Hindi'].apply(lambda x: remove_punctuation(x, language='Hindi'))
    # Removing Contraction
    ds_raw['English'] = ds_raw['English'].apply(expand_contractions)
    # Tokenize
    # nltk.download('punkt')   # ----> Uncomment this line for first time downloading
    ds_raw['English'] = ds_raw['English'].apply(do_tokenization)
    ds_raw['Hindi'] = ds_raw['Hindi'].apply(do_tokenization)
    # Removing stop words
    # nltk.download('stopwords') # ----> Uncomment this line for first time downloading
    # nltk.download('indian') # ----> Uncomment this line for first time downloading
    # Below steps taking longer than expected time. Thus commenting for now
    # ds_raw['English'] = ds_raw['English'].apply(lambda x: remove_stopwords(x, language='english'))
    # ds_raw['Hindi'] = ds_raw['Hindi'].apply(lambda x: remove_stopwords(x, language='hindi'))

    # Stemming Text
    ds_raw['English'] = ds_raw['English'].apply(do_stemming)
    ds_raw['Hindi'] = ds_raw['Hindi'].apply(do_stemming)

    print("After Preprocessing ")
    print("-" * 40)
    # Printing 5 samples
    print(ds_raw.sample(5))

    return ds_raw

def create_hugging_face_dataset(dataframe : pd.DataFrame) :

    data = []
    for index, row in dataframe.iterrows() :
        data.append({"en" : row['English'], "hi" : row['Hindi']})

    dataset = Dataset.from_list(data)
    return dataset

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
        yield item[lang]

def get_or_build_tokenizer(args : argparse.ArgumentParser, ds, lang : str) :

    '''

    Args:
        args:  contains all the argumnets needed in main file ---> For more reference refer to main.py
        ds: dataset are considering ---> ds will generally contain some format where each id from 1 src lang to 1 target lang
        lang: describe for which langauage we are building tokenizer ---> can be src or dest or can be en to it

    Returns: tokenizer where each word is represented with some id ----> e.g. ant ---> 1 the ---> 2 and so on ....

    '''

    args.tokenizer_file = "/data/home/karmpatel/karm_8T/naman/demo/dataset_eng2hi/en2hi_tokenizer_{0}.json"

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

    file_name = '/data/home/karmpatel/karm_8T/naman/demo/dataset_eng2hi/Kaggle_Dataset_English_Hindi.csv'

    dataframe = preprocess_data(file_name)

    # Converting dataset into hugging face form
    dataset = create_hugging_face_dataset(dataframe)

    # Adding some arguments for language
    args.lang_src = "en"
    args.lang_tgt = "hi"

    print(f"Total Train sentences  : {int(len(dataset))}")
    print("Printing sentence details ")
    idx = 100
    print(
        f" \n English : {dataset[idx][args.lang_src]} \n Hindi : {dataset[idx][args.lang_tgt]} ")

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(args, dataset, args.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(args, dataset, args.lang_tgt)

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in dataset:
        src_ids = tokenizer_src.encode(item[args.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item[args.lang_tgt]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of train source sentence: {max_len_src}')
    print(f'Max length of train target sentence: {max_len_tgt}')

    # Defining Sequence Length
    args.seq_len = max(max_len_src, max_len_tgt) + 5

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(dataset))
    val_ds_size = len(dataset) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset2(train_ds_raw, tokenizer_src, tokenizer_tgt, args.lang_src, args.lang_tgt,
                                args.seq_len)
    val_ds = BilingualDataset2(val_ds_raw, tokenizer_src, tokenizer_tgt, args.lang_src, args.lang_tgt,
                              args.seq_len)

    train_dataloader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt












