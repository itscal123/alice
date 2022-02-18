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
from ast import literal_eval
import pickle
from data import convert_sarc_data_to_dataframe

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[LoggingHandler()],
                    level=logging.INFO)


if __name__ == '__main__':

    # This is a pretrained model based on BERT, I'm using it to create our sentence embeddings
    embedder = SentenceTransformer('multi-qa-distilbert-dot-v1')

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
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
            stored_data             = pickle.load(fIn)
            stored_movie_lines      = stored_data['movie_lines']
            stored_sarc_lines       = stored_data['sarc_lines']
            stored_convo_maps       = stored_data['convo_mappings']
            stored_sarc_embeddings  = stored_data['sarc_embeddings']
            stored_movie_embeddings = stored_data['movie_embeddings']

        sarc_embeddings   = stored_sarc_embeddings
        movie_embeddings  = stored_movie_embeddings
        convo_mappings    = stored_convo_maps
        movie_lines       = stored_movie_lines
        sarc_lines        = stored_sarc_lines
    else:
        print('### Pre-processing data ###')
        # based on code from: http://mccormickml.com/2019/07/22/BERT-fine-tuning/#21-download--extract
        ROOT_DIR = os.path.dirname(os.path.abspath(os.curdir))

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
        movie_dialogues_df.lineID = movie_dialogues_df.lineID.apply(lambda x: x.strip())
        movie_dialogues_df.text   = movie_dialogues_df.text.apply(lambda x: x.strip())

        movie_conversations_file = 'data/cornell movie-dialogs corpus/movie_conversations.txt'
        movie_conversations_df = pd.read_csv(movie_conversations_file, index_col=False,
                                         sep=r'\+{3}\$\+{3}',
                                         engine='python',
                                         header=None,
                                         skipinitialspace=True,
                                         encoding='unicode_escape',
                                         na_filter=False,
                                         names=['charID1', 'charID2', 'movieID', 'lineIDs'])

        # strip all whitespace and convert string to list
        # i.e. " ['L000', 'L001', 'L002']" -> ['L000', 'L001', 'L002']
        movie_conversations_df.lineIDs  = movie_conversations_df.lineIDs.apply(lambda x: x.strip())
        movie_conversations_df.lineIDs  = movie_conversations_df.lineIDs.apply(literal_eval)

        sarc_files = ['data/sarcasm_v2/GEN-sarc-notsarc.csv',
                      'data/sarcasm_v2/HYP-sarc-notsarc.csv',
                      'data/sarcasm_v2/RQ-sarc-notsarc.csv']
        sarc_df = pd.concat(convert_sarc_data_to_dataframe(sarc_files), axis=0)
        sarc_lines = np.array(sarc_df.text.values)

        # lines = [[lineID#, text],
        #          ["L0000", "this is an example!"],]
        movie_lines = np.column_stack((movie_dialogues_df.lineID.values, movie_dialogues_df.text.values))

        print('### Encoding corpora ###\n')
        # Start the multi-process pool on all available CUDA devices
        pool = embedder.start_multi_process_pool()

        # Compute the embeddings using the multi-process pool
        movie_embeddings = embedder.encode_multi_process(movie_lines[:,1], pool)
        print("Movie embeddings computed. Shape:", movie_embeddings.shape)

        sarc_embeddings = embedder.encode_multi_process(sarc_lines, pool)
        print("Sarcasm embeddings computed. Shape:", sarc_embeddings.shape)

        # Optional: Stop the processes in the pool
        embedder.stop_multi_process_pool(pool)

        # Store sentences & embeddings on disc
        with open('embeddings.pkl', "wb") as fOut:
            pickle.dump({'movie_lines': movie_lines, 
                         'sarc_lines': sarc_lines, 
                         'convo_mappings': movie_conversations_df.lineIDs, 
                         'sarc_embeddings': sarc_embeddings, 
                         'movie_embeddings': movie_embeddings}, 
                        fOut, 
                        protocol=pickle.HIGHEST_PROTOCOL)

    # https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search.py
    print(movie_embeddings[:5])

    query = ""
    while query != "Quit" and query != "quit" and query != "q":
        query = input('Say something, or type "Quit," "quit," or "q" to quit: ')
        if query == "Quit" or query == "quit" or query == "q":
            break
        print('Calculating response...')

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(5, len(movie_lines))
        query_embedding = embedder.encode(query)
        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, movie_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            print("Most similar line:", movie_lines[:,1][idx], "(Score: {:.4f})".format(score))
            line = movie_lines[:,0][idx]

            next_line = "[N/A]"
            line_filter = movie_conversations_df['lineIDs'].map(lambda u: line in u)
            filtered_df = movie_conversations_df[line_filter]
            convo = filtered_df['lineIDs'].iloc[0]
            if convo.index(line) != len(convo)-1:
                next_line = convo[convo.index(line)+1]
                next_line = movie_dialogues_df[movie_dialogues_df['lineID'] == next_line].text.values[0]
            print("next line:", next_line, "\n")
