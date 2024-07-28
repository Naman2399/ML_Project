import argparse
import os
import random
from pathlib import Path
import tqdm
from tokenizers.implementations import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset.dataset_utils_bert.bert_dataset import BERTDataset


def load_dataset(args : argparse.ArgumentParser) :
    args.max_len = 64 # Why this value is there no particular reason ... Can be modified in future

    # Loading all data into memory
    corpus_movie_conversation = "/data/home/karmpatel/karm_8T/naman/demo/cornell_movie_dialogs_corpus/movie_conversations.txt"
    corpus_movie_lines = "/data/home/karmpatel/karm_8T/naman/demo/cornell_movie_dialogs_corpus/movie_lines.txt"

    with open(corpus_movie_conversation, 'r', encoding='iso-8859-1') as c:
        conv = c.readlines()
    with open(corpus_movie_lines, 'r', encoding='iso-8859-1') as l:
        lines = l.readlines()

    # splitting text using special lines
    lines_dic = {}
    for line in lines:
        objects = line.split(" +++$+++ ")
        lines_dic[objects[0]] = objects[-1]

    # generate question answer pairs
    pairs = []
    for con in conv:
        ids = eval(con.split(" +++$+++ ")[-1])
        for i in range(len(ids)):
            qa_pairs = []

            if i == len(ids) - 1:
                break

            first = lines_dic[ids[i]].strip()
            second = lines_dic[ids[i + 1]].strip()

            qa_pairs.append(' '.join(first.split()[:args.max_len]))
            qa_pairs.append(' '.join(second.split()[:args.max_len]))
            pairs.append(qa_pairs)

    # print a pair example
    example_idx = 20
    print(f"Total pairs :{len(pairs)} ")
    print("Pair Example")
    print(f"Index : {example_idx}")
    print(pairs[20])

    '''
    Tokenization 
    For BERT input always start with [CLS] token 
                          end with   [SEP] token 
    BERT employs WordPiece tokenizer, which can split a single word into multiple tokens.
                E.g.  “surfboarding” is broken down into ['surf', '##boarding', '##ing']
                This technique helps the model to understand that words like surfboardand snowboardhave shared meaning 
                through the common wordpiece ##board
                
                score = (freq_of_pair) / (freq_of_first_element × freq_of_second_element)
    '''

    '''
    Training of BERTWordPieceTokenizer
            1. Saving the conversation text into multiple .txt files (with batch of N=10000)
            2. Some params in tokenizer such as clean text and stripe_accents for chinese chars
            
    '''

    # WordPiece Tokenizer

    # Save data as txt file
    data_dir_path = "/data/home/karmpatel/karm_8T/naman/demo/cornell_movie_dialogs_corpus/data"
    os.makedirs(data_dir_path, exist_ok=True)
    text_data = []
    file_count = 0

    for sample in tqdm.tqdm([x[0] for x in pairs]):
        text_data.append(sample)

        # once we hit the 10K mark, save to file
        if len(text_data) == 10000:
            with open(f'{data_dir_path}/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1

    paths = [str(x) for x in Path(data_dir_path).glob('**/*.txt')]

    # training own tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
    )

    '''
     Tokens special tokens - 
        CLS - serves as the the Start of Sentence (SOS)
        SEP - serves as End of Sentence (EOS) and also the separation token between first and second sentences
        PAD - added into sentences so that all of them would be in equal length
        MASK - word replacement during masked language prediction
        UNK - serves as a replacement for token if it’s not being found in the tokenizer’s vocab
    '''
    tokenizer.train(
        files=paths,
        vocab_size=30_000,
        min_frequency=5,
        limit_alphabet=1000,
        wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )

    # Bert tokenizer path
    bert_tokenizer_path = "/data/home/karmpatel/karm_8T/naman/demo/cornell_movie_dialogs_corpus/bert-it-1"
    bert_tokenizer_prefix_name = "bert-it"
    os.makedirs(bert_tokenizer_path, exist_ok=True)
    tokenizer.save_model(bert_tokenizer_path, bert_tokenizer_prefix_name)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_tokenizer_path, bert_tokenizer_prefix_name+"-vocab.txt"), local_files_only=True)

    train_data = BERTDataset(pairs, seq_len=args.max_len, tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
    sample_data = next(iter(train_loader))

    # Print Sample data
    print(f"Total Train Dataset :{len(train_data)}")
    print("Example -")
    print(train_data[random.randrange(len(train_data))])


    return train_loader, tokenizer