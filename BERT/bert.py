import torch
import os
import pandas as pd
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util, LoggingHandler
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np

# for storing our embeddings
import pickle

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[LoggingHandler()],
                    level=logging.INFO)

from data import convert_sarc_data_to_dataframe

# print("## Loading Tokenizer and Pretrained Model ##")
# # The following initializations are just so I could test out my embed_message function.
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased',
#                                   output_hidden_states=True,  # Whether the model returns all hidden-states.
#                                   )

# commented these lines out because they're preprocessing for training, which we may or may not do
'''
max_length = 512
input_ids = []  # id mappings
attention_masks = []  # padding differentiators
token_type_ids = []  # segment ids

for line in lines:
    encoded_dict = tokenizer.encode_plus(line,
                                         add_special_tokens=True,
                                         max_length=max_length,
                                         padding='max_length',
                                         truncation=True,
                                         return_attention_mask=True,
                                         return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    token_type_ids.append(encoded_dict['token_type_ids'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
token_type_ids = torch.cat(token_type_ids, dim=0)

# create dataset for our data loader
dataset = TensorDataset(input_ids, token_type_ids, attention_masks)

# create data loader to make train loop simpler & more efficient
batch_size = 16
loader = DataLoader(dataset,
                    sampler=RandomSampler(dataset),
                    batch_size=batch_size)
'''

#### Training ####
'''
print("\n## Start Training ##")
epochs = 3
lr = 2e-5

optimizer = AdamW(model.parameters(),
                  lr=lr,
                  eps=1e-8)

# loop from here: https://towardsdatascience.com/how-to-train-bert-aaad00533168
for epoch in range(epochs):
    loop = tqdm(loader)
    model.train()  # put model in a training state
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optimizer.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch[0].to(device)
        token_type_ids = batch[1].to(device)
        attention_mask = batch[2].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        # extract loss
        loss = outputs.loss

        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optimizer.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
'''

if __name__ == '__main__':

    # This is a pretrained model based on BERT, I'm using it to create our sentence embeddings
    embedder = SentenceTransformer('multi-qa-distilbert-dot-v1')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    embedder.to(device)

    # The following code is to make sentence embeddings faster
    # https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/computing-embeddings
    # /computing_embeddings_mutli_gpu.py Important, you need to shield your code with if __name__. Otherwise,
    # CUDA runs into issues when spawning new processes.

    print('### Finding embeddings... ###')
    pickled_embs_file = 'embeddings.pkl'
    if os.path.exists(pickled_embs_file):
        # Load sentences & embeddings from disc
        with open('embeddings.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_sentences = stored_data['sentences']
            stored_embeddings = stored_data['embeddings']

        corpus_embeddings = stored_embeddings
        lines = stored_sentences
    else:
        print('#### Pre-processing data ####')
        # based on code from: http://mccormickml.com/2019/07/22/BERT-fine-tuning/#21-download--extract
        ROOT_DIR = os.path.dirname(os.path.abspath(os.curdir))

        print("\n## Preprocessing Data ##")
        # load datasets into dataframes
        movie_dialogues_file = 'data/cornell movie-dialogs corpus/movie_lines.txt'
        # can comment this line out later, i just need it for me
        movie_dialogues_file = os.path.join(ROOT_DIR, movie_dialogues_file)
        movie_dialogues_df = pd.read_csv(movie_dialogues_file, index_col=False,
                                         sep=r'\+{3}\$\+{3}',
                                         engine='python',
                                         header=None,
                                         skipinitialspace=True,
                                         encoding='unicode_escape',
                                         na_filter=False,
                                         names=['lineID', 'charID', 'movieID', 'charName', 'text'])
        # print('Num sentences: {:,}\n'.format(movie_dialogues_df.shape[0]))
        # print(movie_dialogues_df.sample(10))

        sarc_files = ['data/sarcasm_v2/GEN-sarc-notsarc.csv',
                      'data/sarcasm_v2/HYP-sarc-notsarc.csv',
                      'data/sarcasm_v2/RQ-sarc-notsarc.csv']
        sarc_dfs = convert_sarc_data_to_dataframe(sarc_files)
        # for df in sarc_dfs:
        #     print('Num sentences: {:,}\n'.format(df.shape[0]))
        #     print(df.sample(10))

        lines = np.array([])
        for df in sarc_dfs:
            lines = np.concatenate((lines, df.text.values), axis=None)
        lines = np.concatenate((lines, movie_dialogues_df.text.values), axis=None)
        # print(len(lines)) #309,406 lines
        # print('First 5 lines: {}'.format(lines[:5]))

        print('### Encoding corpora ###\n')
        # for now all of our sentence embeddings will be stored in memory, we'll see how heavy memory usage is

        # Start the multi-process pool on all available CUDA devices
        pool = embedder.start_multi_process_pool()

        # Compute the embeddings using the multi-process pool
        corpus_embeddings = embedder.encode_multi_process(lines, pool)
        print("Embeddings computed. Shape:", corpus_embeddings.shape)

        # Optional: Stop the processes in the pool
        embedder.stop_multi_process_pool(pool)

        # Store sentences & embeddings on disc
        with open('embeddings.pkl', "wb") as fOut:
            pickle.dump({'sentences': lines, 'embeddings': corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    # https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search.py
    print(corpus_embeddings[:5])
    query = input('Query: ')
    print('Calculating response...')

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(lines))
    query_embedding = embedder.encode(query)
    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(lines[idx], "(Score: {:.4f})".format(score))

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """
