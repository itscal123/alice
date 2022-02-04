from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

MAX_LENGTH = 10
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class Encoder(nn.Module):
    """
    Bidirectional GRU encoder for the general chatbot trained on the PolyAI
    reddit data.
    params: 
        input_seq: batch of input sentences; shape = (max_length, batch_size)
        
        input_lengths: list of sentence lengths corresponding to each sentence in the batch; shape = (batch_size)
        
        hidden: hidden state; shape (n_layers * num_directions, batch_size, hidden_size)

    returns:
        outputs: output features from last hidden layer of the GRU (sum of bidirectional outputs); shape = (max_len, batch_size, hidden_size)
        
        hidden: updated hidden state from GRU; shape = (n_layers * num_directions, batch_size, hidden_size)
    """
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.n_layers
        self.hidden_size
        self.embedding
        # Initialize GRU. input_size and hidden_size params are set to hidden_size
        # because input_size is a word embedding with number of features
        # equal to the hidden_size

        # TODO: additional GRU hidden layers???
        
        self.gru = nn.GRU(hidden_size,
                          hidden_size, 
                          n_layers, 
                          dropout=(0 if n_layers==1 else dropout),
                          bidirectional=True)
        

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert indices to word embeddings
        embedded = self.embedding(input_seq)
        # Pack padded sequences
        packed = nn.utils.pack_padded_sequence(embedded, input_lengths)
        # Forward pass
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.pad_packed_sequence(outputs)
        # Sum bidirectional GRUs (left-to-right and right-to-left)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Returns output and final hidden state
        return outputs, hidden
        

class GlobalAttn(nn.Module):
    """
    Implementation of Luong's Global Attention mechanism which calculates the 
    the attention energies bewteen the encoder output and decoder output
    which are known as score functions.
    Outputs a softmax normalized weights tensor of shape (batch_size, 1, max_length)
    """
    def __init__(self, method, hidden_size):
        super(GlobalAttn, self).__init__()
        self.method = method
        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, "is not an appropriate Attention method")
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))


    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)


    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)


    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)


    def forward(self, hidden, encoder_outputs):
        # Calculate Attention weights based off method
        if self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_score(hidden, encoder_outputs)
        else:
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_len and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return softmax probabilities (normalized) with added dimension
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class Decoder(nn.Module):
    """
    Unidirectional GRU model decoder for the general chatbot trained with Global 
    Attention. Manually feed each batch one time step (i.e., one word) at a time.
    Consequently, the embedded word tensor and GRU output both have shape (1, batch_size, hidden_size)
    params:
        input_step: one time step (one word) of input sequence batch; shape = (1, batch_size)

        last_hidden: final hidden layer of GRU; shape = (n_layers * num_directions, batch_size, hidden_size)

        encoder_outputs: encoder's outputs; shape (n_layers * num_directions, batch_size, hidden_size)

    returns:
        output: softmax normalized tensor giving probabilities of each word being the correct next word in the decoded sequence;
                shape = (batch_size, voc.num_words)

        hidden: final hidden state of GRU; shape = (n_layers * num_directions, batch_size, hidden_size)
    """
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        # Store parameters
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        # TODO: Additional hidden GRU layers???
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = GlobalAttn(attn_model, hidden_size)


    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: run one word (time step) at a time
        # Retreive embedding for current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # Forward pass through unidirectional GRU
        gru_output, hidden = self.gru(embedded, last_hidden)

        # Calculate current Attention weights from current GRU output and encoder outputs
        attn_weights = self.attn(gru_output, encoder_outputs)

        # Multiply Attention weights with encoder_outputs to get weighted sum context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))

        # Concatenate weighted vector context and GRU output
        gru_output = gru_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((gru_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Predict next word
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        # Return output and final hidden state
        return output, hidden


def maskNLL(inp, target, mask):
    """
    Custom loss function for model training. Since the inputs are batches of padded
    sequences, we cannot consider all elements in the input tensor. Instead, calculate
    loss by decoder's output tensor, target tensor, and a binary mask tensor that describes
    the padded target tensor. Negative log likelihood of the elements that are are masked
    (i.e., marked 1 in the mask tensor).
    """
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, 
          encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, 
          batch_size, clip, max_length=MAX_LENGTH):
    """
    Custom training function for a single training iteration on a batch of inputs. 
    Incorporates both gradient clipping and teacher forcing. Gradient clipping is a 
    technique to deal with the vanishing/exploding gradient problem that can arise 
    in neural network training (especially in the RNN framework). Teacher forcing
    is another technique especially helpful in difficult learning tasks like NLP which promote
    faster convergence during training by allowing the decoder to occassionaly access 
    the target word as part of its next word prediction
    """
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Lengths for the rnn packing should be done on cpu
    lengths = lengths.to("cpu")

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS token for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Check if we are using Teacher Forcing on this iteration
    use_teacher_forcing = True if random.random < teacher_forcing_ratio else False

    # Forward pass through decoder one time step at a time 
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # Teacher Forcing -> Next input is current target word
            decoder_input = target_variable[t].view(1, -1)

            # Calculate/Accumulate oss
            mask_loss, nTotal = maskNLL(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # No Teacher Forcing -> Next input is decoder's current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

            # Calvulate/Accumulate loss
            mask_loss, nTotal = maskNLL(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropagation
    loss.backward()

    # Clip gradients; gradients are modified in-place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                   print_every, save_every, clip, corpus_name, loadFilename):
    """
    Trains the model for n_iterations. Saving the model involves saving a tarball containing the 
    encoder and decoder state_dicts (parameters), optimizers' state_dicts, loss, iteration, etc. 
    Provides flexibility with checkpoint. After laoding a checkpoint, we can use the model parameters
    to run inference, or we can continue to train where we left off.
    """
    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]

    # Initializations
    print("Initializing...")
    start_iteration = 1
    print_loss = 0
    if loadFileName:
        start_iteration = checkpoint["iteration"] + 1
    
    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration+1):
        training_batch = training_batch[iteration-1]

        # Extract fields from batches
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, 
                        encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; "
                    "Percent complete: {:.1f}%; "
                    "Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every==0):
            directory = os.path.join(save_dir, model_name, corpus_name, "{}-{}_{}".format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedires(directory)
            torch.save({
                "iteration": iteration,
                "en": encoder.state_dict(),
                "de": decoder.state_dict(),
                "en_opt": encoder_optimizer.state_dict(),
                "de_opt": decoder_optimizer.state_dict(),
                "loss": loss,
                "voc_dict": voc.__dict__,
                "embedding": embedding.state_dict()
            }, os.path.join(directory, "{}_{}.tar".format(iteration, "checkpoint")))


# greedy decoding implementation; replace with beam search later
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward pass through encoder
        encoder_ouputs, encoder_hidden = self.encoder(input_seq, input_length)

        # Prepare encoder's final hidden layer to be first hidden layer to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # Initialize decoder with SOS token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token

        # Initialized tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        # Iteratively decode one wor token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)

            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        # Return collections of word tokens and scores
        return all_tokens, all_scores


# Beam search implementation; todo
class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]

    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0,1)
    
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")

    # Decoder senetence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)

    # Indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


# TODO: Change to generator later???
def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ""
    while(True):
        try:
            # Get input sentence
            input_sentence = input("> ")

            # Check if it is quit case
            if input_sentence == "q" or input_sentence == "quit":
                break

            # Normalize sentence
            input_sentence = normalizeString(input_sentence)

            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)

            # Format and print reponse
            output_words[:] = [x for x in output_words if not (x == "EOS" or x == "PAD")]
            print("Bot:", " ".join(output_words))
        
        except KeyError():
            print("Error: Encountered unknown word")


if __name__ == "__main__":
    # TODO: data preprocessing
    # TODO: train and save model either with pickle or json
    # TODO: implement beam search for decoder output
    # Configure models
    model_name = "generative_model"
    attn_model = "dot"
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load; None if training for first time
    loadFileName = None
    checkpoint_iter = 4000
    #loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                            '{}_checkpoint.tar'.format(checkpoint_iter))

    # Load model if a loadFilename is provided
    if loadFileName:
        # If loading on same machine model was trained on
        checkpoint = torch.laod(loadFileName)
        # If loading model to CPU
        # checkpoint = torch.load(loadFileName, map_location=torch.device("cpu"))
        encoder_sd = checkpoint["en"]
        decoder_sd = checkpoint["de"]
        encoder_optimizer_sd = checkpoint["en_opt"]
        decoder_optimizer_sd = checkpoint["de_opt"]
        embeddig_sd = checkpoint["embedding"]
        voc.__dict__ = checkpoint["voc_dict"]

    print("Building encoder and decoder...")
    # Initialzie word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFileName:
        embedding.load_state_dict(embedding_sd)
    
    # Initialize encoder and decoder models
    encoder = Encoder(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = Decoder(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFileName:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print("Models built and ready to train")

    # Configure training/optimization
    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 4000
    print_every = 1
    save_every = 500

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print("Building optimizers...")
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    if loadFileName:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # If CUDA available
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Train
    print("Training...")
    trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding,
               encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, 
               clip, corpus_name, loadFilename)
    print("Training Completed")

    # Evaluation
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)
    #searcher = BeamSearchDecoder(encoder, decoder)

    # Begin chat (uncomment)
    # evaluateInput(encoder, decoder, searcher, voc)