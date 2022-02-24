import torch
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util, LoggingHandler
import numpy as np
from ast import literal_eval
import pickle
from data_processor import convert_sarc_data_to_dataframe

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[LoggingHandler()],
                    level=logging.INFO)

# STATIC FUNCTIONS

# if we ever wanted to load the embeddings into something else without instantiating the class, or loading the model
def load_embeddings():
    """ Loads the embeddings, then returns a dictionary with entries as the different files """
    pickled_embs_file = 'embeddings.pkl'
    embeddings = dict()
    if os.path.exists(pickled_embs_file):
        # Load sentences & embeddings from disc
        with open('embeddings.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_movie_lines = stored_data['movie_lines']
            stored_sarc_lines = stored_data['sarc_lines']
            stored_convo_maps = stored_data['convo_mappings']
            stored_sarc_embeddings = stored_data['sarc_embeddings']
            stored_movie_embeddings = stored_data['movie_embeddings']

        embeddings['sarc_embeddings'] = stored_sarc_embeddings
        embeddings['movie_embeddings'] = stored_movie_embeddings
        embeddings['convo_mappings'] = stored_convo_maps
        embeddings['movie_lines'] = stored_movie_lines
        embeddings['sarc_lines'] = stored_sarc_lines
    else:
        print("Embeddings don't exist! Please run bert.py")
    return embeddings


class BertAlice:
    """ Hugging face SentenceTransformer model to generate sentence embeddings and output a response to a query
        calculated by the argmax of the dot product with pre-computed embeddings. Additionally, our model is fine-tuned
        for Sentence Classification on sarcastic versus non-sarcastic sentences.
        References: 1) https://aajanki.github.io/fi-sentence-embeddings-eval/models.html for basic architecture
                    1.5) https://www.sbert.net/docs/training/overview.html for architecture and training
                    2) https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/other/training_multi-task.py
                        for multitask training
                    3) https://www.sbert.net/examples/applications/semantic-search/README.html
                        https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search.py
                        for retrieving responses based on semantic similarity/search
                    4) https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/#:~:text=It%20is%20designed%20to%20pre,wide%20range%20of%20NLP%20tasks.%E2%80%9D
                        https://github.com/huggingface/notebooks/blob/master/examples/text_classification.ipynb
                        https://huggingface.co/docs/transformers/training
                        for fine-tuning
                    """

    def __init__(self, model_name='multi-qa-distilbert-dot-v1', device='cpu'):
        """ Loads the correct model.
            :parameter model_name Name of the model we want to load, it is set to a default because we've already
            calculated the embeddings from this model. """
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = load_embeddings()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(device)

    def __repr__(self):
        # TODO
        pass

    def train(self, data, max_seq_length):
        """ Training function for our model. We will be using this to fine-tune the base model for our downstream task.
            The goal is to keep adjusting our model weights to distinguish more humorous/sarcastic sentences versus
            non-sarcastic ones. e.g. Sentence Classification
            :parameter data Data that we will be using to train on. Includes sentences and their labels.
            :parameter max_seq_length The length at which we will truncate our tokenized inputs because BERT needs
                                        normalized vector lengths. """
        # TODO
        # I think we should do multitask training, so a bunch of preprocessing will go here
        # https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/other/training_multi-task.py
        pass

    def get_response(self, query: str):
        """ Run a semantic search against our query, then output the following line from our corpus of movie dialogue.
            :parameter query The user query """
        # TODO
        query_embedding = self._encode(query)
        # compare query embedding to movie embeddings via cosine similarity
        cos_scores = util.cos_sim(query_embedding, self.embeddings['movie_embeddings'])[0]
        # get 5 most similar sentences
        # topk returns a named tuple of Tensors: ('values', 'indices')
        # in our case, the tuple will consist of two Tensors of size 5
        potential_responses = []
        top_results = torch.topk(cos_scores, 5)
        # for each similar sentence, add the next line and it's score (cosine-similarity to the user's query) to the list of
        # potential responses if available (not [N/A])
        for score, idx in zip(top_results[0], top_results[1]):
            next_line = self._get_next_line(self.embeddings['movie_lines'][:, 0][idx])
            if next_line[1] != "[N/A]":
                cos_score = util.cos_sim(query_embedding, self.embeddings['movie_embeddings'][next_line[0]])[0]
                potential_responses.append((cos_score,next_line[1]))

        # return the response with the highest score
        return max(potential_responses, key=lambda a:a[0])

        

    # HELPER FUNCTIONS

    def _encode(self, query: str):
        """ Input is a query string which we will encode through our model. The output is our query embedding. """
        return self.model.encode(query, show_progress_bar=False)

    def _get_next_line(self, line_id):
        """ Outputs the following line of dialogue given a line id (e.g. 'L0000') 
            from the movie lines corpus.
            :parameter line_id is the line id of the preceding line -- the one most similar to
            the user's query.
            :return a tuple containing the string with the next line and its corresponding index in movie_embeddings
            e.g., (12345, "why hello there") """

        # return [N/A] if there is no following line.
        next_line = "[N/A]" 
        next_line_idx = -1
        # filter out all rows that do not contain line_id (there should only be 1 row containing line_id)
        line_filter = self.embeddings['convo_mappings'].map(lambda u: line_id in u)
        filtered_df = self.embeddings['convo_mappings'][line_filter]
        # convo is a pandas df with 1 entry and NO columns names. the entry is a list of line_ids.
        # e.g. [['L000', 'L001', 'L002']]
        convo = filtered_df.iloc[0]
        # if the given line_id isn't the last line in a conversation, get the next line's line_id
        # and get the corresponding line from self.embeddings['movie_lines'] using np.where
        if convo.index(line_id) != len(convo) - 1:
            next_line = convo[convo.index(line_id) + 1]  # e.g. 'L001'
            next_line_idx = np.where(self.embeddings['movie_lines'][:, 0] == next_line)[0][0]
            next_line = self.embeddings['movie_lines'][:, 1][np.where(self.embeddings['movie_lines'][:, 0] == next_line)][0]
        return (next_line_idx, next_line)

    # EXTRA FUNCTIONS

    def get_topk_similar(self, k, query, corpus_embedding='movie_embeddings'):
        """ Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
            put this into the class for funzies... idk if we will actually need it
            :param query string
            :param int of how many similar results to search for
            :param corpus_embedding which corpus we want to compare embeddings against
            :return Union of tensors with the top k most similar results """

        top_k = min(k, len(self.embeddings[corpus_embedding]))
        query_embedding = self._encode(query)
        # We use cosine-similarity and torch.topk to find the highest k scores
        cos_scores = util.cos_sim(query_embedding, self.embeddings[corpus_embedding])[0]
        # topk returns named tuples of the results as a named tuple of tensors 'values' and 'indices'
        return torch.topk(cos_scores, k=top_k)

    def get_sarcastic_responses(self, query: str, k: int):
        """ Prints sarcastic responses for a query
            params: query = string
                    k = int of how many results to output """
        top_results = self.get_topk_similar(k, query)
        ranking = 1
        # top_results is a tuple of tensors: ('values': Tensor, 'indices': Tensor)
        for score, idx in zip(top_results[0], top_results[1]):
            line = self.embeddings['movie_lines'][:, 1][idx]
            line_id = self.embeddings['movie_lines'][:, 0][idx]

            print(str(ranking) + ') ' + line + " (Score: {:.4f})".format(score) + '\n')

            next_line = self._get_next_line(line_id)
            print("   Corresponding response: " + next_line + '\n')

            # if there exists a response to the most similar line (to the user's query),
            # then find text from the sarcasm corpus that is similar to that response.
            if next_line != "[N/A]":
                print("   Sarcastic responses:\n")
                top_sarcastic_results = self.get_topk_similar(5, next_line, 'sarc_embeddings')
                for s, i in zip(top_sarcastic_results[0], top_sarcastic_results[1]):
                    print("\t- " + self.embeddings['sarc_lines'][i].strip() + " (Score: {:.4f})".format(s))
            else:
                print("    No sarcastic response found. Outputting last line instead:\n")
                print(line + '\n')
            ranking += 1
            print()


if __name__ == '__main__':

    pickled_embs_file = 'embeddings.pkl'
    if not os.path.exists(pickled_embs_file):
        print('### Pre-processing data ###')
        # This is a pretrained model based on BERT, I'm using it to create our sentence embeddings
        embedder = SentenceTransformer('multi-qa-distilbert-dot-v1')
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")
        embedder.to(device)

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
        movie_dialogues_df.text = movie_dialogues_df.text.apply(lambda x: x.strip())

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

        movie_conversations_df.lineIDs = movie_conversations_df.lineIDs.apply(lambda x: x.strip())
        movie_conversations_df.lineIDs = movie_conversations_df.lineIDs.apply(literal_eval)
        convo_mappings = movie_conversations_df.lineIDs

        sarc_files = ['data/sarcasm_v2/GEN-sarc-notsarc.csv',
                      'data/sarcasm_v2/HYP-sarc-notsarc.csv',
                      'data/sarcasm_v2/RQ-sarc-notsarc.csv']
        sarc_df = pd.concat(convert_sarc_data_to_dataframe(sarc_files), axis=0)
        sarc_lines = np.array(sarc_df.text.values)

        # lines = [[lineID#, text],
        #          ["L0000", "this is an example!"],]
        movie_lines = np.column_stack((movie_dialogues_df.lineID.values, movie_dialogues_df.text.values))

        # The following code is to make sentence embeddings faster
        # https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/computing-embeddings
        # /computing_embeddings_mutli_gpu.py Important, you need to shield your code with if __name__. Otherwise,
        # CUDA runs into issues when spawning new processes.

        print('### Encoding corpora ###\n')
        # Start the multi-process pool on all available CUDA devices
        pool = embedder.start_multi_process_pool()

        # Compute the embeddings using the multi-process pool
        movie_embeddings = embedder.encode_multi_process(movie_lines[:, 1], pool)
        print("Movie embeddings computed. Shape:", movie_embeddings.shape)

        sarc_embeddings = embedder.encode_multi_process(sarc_lines, pool)
        print("Sarcasm embeddings computed. Shape:", sarc_embeddings.shape)

        # Optional: Stop the processes in the pool
        embedder.stop_multi_process_pool(pool)

        # Store sentences & embeddings on disc
        with open('embeddings.pkl', "wb") as fOut:
            pickle.dump({'movie_lines': movie_lines,
                         'sarc_lines': sarc_lines,
                         'convo_mappings': convo_mappings,
                         'sarc_embeddings': sarc_embeddings,
                         'movie_embeddings': movie_embeddings},
                        fOut,
                        protocol=pickle.HIGHEST_PROTOCOL)

        print("Sample of the movie embeddings:\n")
        print(movie_embeddings[:5])

    demo = input("Would you like to demo BertAlice? [y/n]: ")
    if demo == 'y':
        bert = BertAlice()
        query = ""
        while query != "Quit" and query != "quit" and query != "q":
            query = input('Say something, or type "Quit," "quit," or "q" to quit: ')
            if query == "Quit" or query == "quit" or query == "q":
                break
            print('Calculating response...')
            print("\n\n======================\n\n")
            print("Query:", query)
            print("Response:", bert.get_response(query))
            # print("\nTop 5 most similar sentences in corpus:")
            # bert.get_sarcastic_responses(query, 5)