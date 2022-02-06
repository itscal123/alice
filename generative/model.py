from __future__ import unicode_literals

import torch
import pickle
from train import Encoder, Decoder, GreedySearchDecoder, GlobalAttn
from data import Voc, normalizeString, indexesFromSentence, Data
import unicodedata
import codecs

class Generative():
    def __init__(self):
        self.encoder = torch.load("generative\models\encoder.pt")
        self.decoder = torch.load("generative\models\decoder.pt")
        self.searcher = torch.load("generative\models\searcher.pt")
        data = pickle.load(open("generative\data.p", "rb"))
        voc, pairs, save_dir, corpus_name = data.loadData()
        self.voc = voc
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")


    def evaluate(self, sentence, max_length=10):
        # Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(self.voc, sentence)]

        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0,1)
        
        # Use appropriate device
        input_batch = input_batch.to(self.device)
        lengths = lengths.to("cpu")

        # Decoder sentence with searcher
        tokens, scores = self.searcher(input_batch, lengths, max_length)

        # Indexes -> words
        decoded_words = [self.voc.index2word[token.item()] for token in tokens]
        return decoded_words

    def run(self):
        input_sentence = ""
        while(True):
            try:
                # Get input sentence
                input_sentence = input("User: ")

                # Check if it is quit case
                if input_sentence == "q" or input_sentence == "quit":
                    break

                # Normalize sentence
                input_sentence = normalizeString(input_sentence)

                # Evaluate sentence
                output_words = self.evaluate(input_sentence)

                # Format and print reponse
                output_words[:] = [x for x in output_words if not (x == "EOS" or x == "PAD")]
                print("ALICE:", " ".join(output_words))
            
            except KeyError as e:
                print("Sorry I've never heard the word {} before.".format(e.args[0]))
        print("ALICE: See you next time!")

if __name__ == "__main__":
    model = Generative()
    model.run()