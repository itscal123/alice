# TODO: Interface class that should handle user inputs and model outputs
# TODO: Load the IR model class
import sys
sys.path.insert(1, "generative\\")  # Add generative folder to path

# Generative model imports
from model import Generative
from train import Encoder, Decoder, GreedySearchDecoder, GlobalAttn, BeamSearchDecoder
from data import Voc, normalizeString, indexesFromSentence, Data

if __name__ == "__main__":
    generative = Generative()