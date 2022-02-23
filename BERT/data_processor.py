import pandas as pd
import torch
import os
from transformers import BertTokenizer
from nltk.tokenize import RegexpTokenizer
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(os.curdir))


def convert_sarc_data_to_dataframe(paths: [str]):
    """ Reads sarc datasets into csv file """
    dfs = []
    for path in paths:
        # this line didn't work for me. since we have the data in the github project dir,
        # i think we can just reference it 'locally' -- without the root path prefix.
        # yes, we can, but for some reason it doesn't recognize the reference for me, so i'm uncommenting for now
        # file = os.path.join(ROOT_DIR, path)
        df = pd.read_csv(path, index_col='id')
        # df = pd.read_csv(file, index_col='id')
        df = df[df['class'] == 'sarc']  # only use sarcastic text
        # print('Num sentences in {}: {:,}\n'.format(path, df.shape[0]))
        # print('Samples:\n {}\n'.format(df.sample(3)))
        dfs.append(df)
    return dfs


def process_sarc_data_for_training():
    """ This function is pretty much the same as the one above, but we keep the labels
    :return: dataframes with text and labels from sarcasm corpus
    """
    # load dataset into a dataframe
    sarc_files = ['data/sarcasm_v2/GEN-sarc-notsarc.csv',
                  'data/sarcasm_v2/HYP-sarc-notsarc.csv',
                  'data/sarcasm_v2/RQ-sarc-notsarc.csv']
    dfs = []

    for path in sarc_files:
        path = os.path.join(ROOT_DIR, path)
        df = pd.read_csv(path, index_col='id')

        # print('Num sentences: {:,}\n'.format(df.shape[0]))
        # print(df.sample(10))

        # lines = df.text.values
        # print('First 5 lines: {}'.format(lines[:5]))

        dfs.append(df)

    # print('Loading BERT tokenizer...')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #
    # # comparing sentences
    # print('Original: ', lines[9])
    # print('Tokenized: ', tokenizer.tokenize(lines[9]))
    # print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(lines[9])))
    return dfs


### from https://github.com/Nielspace/BERT/blob/master/BERT%20Text%20Classification%20fine-tuning.ipynb
# with modifications for just word embeddings
class DATALoader:
    def __init__(self, data, max_length):
        self.data = data
        self.tokeniser = BertTokenizer.from_pretrained('bert-base-cased')
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = str(self.data[item])
        data = " ".join(data.split())

        inputs = self.tokeniser.encode_plus(
            data,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )

        ids = inputs["input_ids"]
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_length - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }



def getDataStatistics(df):
    """ Expects a df with 1 column of strings (sentences)"""
    tokenizer = RegexpTokenizer(r'\w+')
    df['tokenized_sents'] = df.apply(tokenizer.tokenize)
    df['sents_length'] = df['tokenized_sents'].apply(lambda row: len(row))
    # print(df['tokenized_sents'].values)
    # print(df['sents_length'].values)
    x = np.array(df['sents_length'].values)
    print("Total number of words:", x.sum(),
          "\nMax/Min doc length:   ", x.max(), '/', x.min(),
          "\nAverage words per doc:", np.average(x), '\n')

if __name__ == "__main__":
    movie_dialogues_file = 'data/cornell movie-dialogs corpus/movie_lines.txt'
    movie_dialogues_df = pd.read_csv(movie_dialogues_file, index_col=False,
                                        sep=r'\+{3}\$\+{3}',
                                        engine='python',
                                        header=None,
                                        skipinitialspace=True,
                                        encoding='unicode_escape',
                                        na_filter=False,
                                        names=['lineID', 'charID', 'movieID', 'charName', 'text'])

    # strip all whitespace
    movie_dialogues_df.text = movie_dialogues_df.text.apply(lambda x: x.strip())   


    sarc_files = ['data/sarcasm_v2/GEN-sarc-notsarc.csv',
                  'data/sarcasm_v2/HYP-sarc-notsarc.csv',
                  'data/sarcasm_v2/RQ-sarc-notsarc.csv']
    sarc_df = pd.concat(convert_sarc_data_to_dataframe(sarc_files), axis=0)

    getDataStatistics(movie_dialogues_df.text)
    getDataStatistics(sarc_df.text)

    # process_sarc_data_for_training()
