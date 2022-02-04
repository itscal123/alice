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