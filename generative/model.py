from __future__ import unicode_literals

import torch
import pickle
from train import Encoder, Decoder, GreedySearchDecoder, GlobalAttn, BeamSearchDecoder
from data import Voc, normalizeString, indexesFromSentence, Data
import unicodedata
import codecs

class Generative():
    def __init__(self):
        self.encoder = torch.load("generative\models\encoder.pt")
        self.decoder = torch.load("generative\models\decoder.pt")
        self.beamSearcher = BeamSearchDecoder(self.encoder, self.decoder, 10)
        data = pickle.load(open("generative\data.p", "rb"))
        voc, pairs, save_dir, corpus_name = data.loadData()
        self.voc = voc
        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if USE_CUDA else "cpu")


    def evaluate(self, sentence, max_length=8):
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

        # Decode sentence with Greedy Searcher
        #tokens, scores = self.greedy(input_batch, lengths, max_length)

        # Indexes -> words
        #decoded_words = [self.voc.index2word[token.item()] for token in tokens]

        # Decode sentence with Beam Search 
        beamTokens = self.beamSearcher(input_batch, lengths, max_length)

        # indexes to words
        decoded_words = [self.voc.index2word[token.item()] for token in beamTokens]

        # Remove start of sentence, end of sentence, and padding tokens
        decoded_words[:] = [x for x in decoded_words if not (x == "SOS" or x == "EOS" or x == "PAD")]

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
                print("ALICE:", " ".join(output_words))
            
            except KeyError as e:
                print("Sorry I've never heard the word {} before.".format(e.args[0]))
        print("ALICE: See you next time!")


    def response(self, input_sentence):
        try:
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)

            # Evaluate sentence
            output_words = self.evaluate(input_sentence)


            return " ".join(output_words)
        
        except KeyError as e:
            return "Sorry I've never heard the word {} before.".format(e.args[0])


if __name__ == "__main__":
    model = Generative()