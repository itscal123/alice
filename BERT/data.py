import pandas as pd
import torch
from transformers import BertTokenizer

# load dataset into a dataframe
# file = '/Users/bianca/Documents/SCHOOL/CS175/alice/data/cornell movie-dialogs corpus/movie_lines.txt'
#
# df = pd.read_csv(file, index_col=False,
#                  sep=r'\+{3}\$\+{3}',
#                  engine='python',
#                  header=None,
#                  skipinitialspace=True,
#                  encoding='unicode_escape',
#                  na_filter=False,
#                  names=['lineID', 'charID', 'movieID', 'charName', 'text'])
#
# print('Num sentences: {:,}\n'.format(df.shape[0]))
# print(df.sample(10))
#
# lines = df.text.values
# print('First 5 lines: {}'.format(lines[:5]))
#
# print('Loading BERT tokenizer...')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# #comparing sentences
# print('Original: ', lines[9])
# print('Tokenized: ', tokenizer.tokenize(lines[9]))
# print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(lines[9])))

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
### END

#some text has len > ~670, we'll truncate those
# loaded_data = DATALoader(lines, 150)
# # look at 10th input
# sample = loaded_data[9]
#
# print("Token IDs:", sample["ids"])
# print("Segment IDs", sample["token_type_ids"]) # token_type_ids = segment_ids
