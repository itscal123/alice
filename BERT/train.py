# module for training a custom text classification model which we will use as our embedder in bert.py
import datetime
import time
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import transformers
from data_processor import process_sarc_data_for_training

# OVERALL TODO
# These are just notes for my reference -- read at risk of confusion:
# I want to fine-tune the model on text classification using the sarcasm dataset
# There are many ways to go about this:
# 1) Using the huggingface doc for how to fine-tune pretrained BERT
# 2) Use sbert doc for tuning for specific downstream task or even multitask train
# 3) Use template how-to code
# Then, save the model, and use as the base model for our sentence transformer (class defined below)
# This class will be the one used for the interface, it will have functions for embedding the query and retrieving
# a response

# reference for training code:
# https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/#:~:text=It%20is%20designed%20to%20pre,wide%20range%20of%20NLP%20tasks.%E2%80%9D


# specify GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load data into dataframe
df = pd.concat(process_sarc_data_for_training(), axis=0)
# let's look at a sample
print(df.sample(10))

# our labels need to be numeric, so let's replace that now
# [0] for notsarc and [1] for sarc
df['class'] = df['class'].replace(['notsarc', 'sarc'], [0, 1])

# split train dataset into train, validation and test sets
# using the sklearn package for splitting
train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['class'],
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=df['class'])

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

# we will now fine-tune the model using the train set and validation set, and make predictions for the test set

# load the model
# we need to specify that we want the hidden states to be outputted
config = AutoConfig.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
bert = AutoModel.from_pretrained('distilbert-base-uncased', config=config)

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# now we process the data for training

# this includes:
#   - tokenizing the data
#   - adding special tokens
#   - padding/truncating text to be of fixed length

# let's first take a look at the length distribution of our data

# get length of all the messages in the train set
seq_len = [len(i.split()) for i in train_text]

# plot the distribution
series = pd.Series(seq_len).hist(bins=30)
plt.show()  # we can see that most of our data is around length 20, but it extends up to 140

# we'll pad/truncate our sentences to 100
# it will make training slower, but that's ok for our purposes

# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length=100,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length=100,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length=100,
    pad_to_max_length=True,
    truncation=True
)

# our data is now encoded and the lengths normalized
# let's convert the lists to tensors

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

# now lets put our data into a dataloader, for simplicity when loading our data when training

# define a batch size (recommended 16 or 32)
batch_size = 16

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)


# now let's define our model architecture
class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 2)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x


# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

# we will use AdamW as our optimizer
optimizer = AdamW(model.parameters(),
                  lr=1e-5)  # learning rate

# now we'll balance out our weights in case of having uneven amounts of sarc - notsarc

# compute the class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)

print("Class Weights:", class_weights)

# now we'll define our loss with the weights

# converting list of class weights to a tensor
weights = torch.tensor(class_weights, dtype=torch.float)

# push to GPU
weights = weights.to(device)

# define the loss function
cross_entropy = nn.NLLLoss(weight=weights)

# number of training epochs
epochs = 10


# now that we've done our data preprocessing, data loading, defined our models with the optimizer, loss functions
# and weights, we can start training.
# this entails:
#   - creating a train function to train the model
#   - creating an evaluator function to evaluate our results

# first we'll create a train function
def train():
    # put our model into a training state
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# function for evaluating the model
def evaluate():
    print("\nEvaluating...")
    t0 = time.time()

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


# now we can finally start fine-tuning the model

# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses = []
valid_losses = []

# for each epoch
for epoch in range(epochs):

    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    # train model
    train_loss, _ = train()

    # evaluate model
    valid_loss, _ = evaluate()

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

# load weights of best model
# they were saved during training
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))

# once we've loaded the weights, we can use the newly fine-tuned model to make predictions on the test set

# get predictions for test data
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

# let's look at the model's performance
preds = np.argmax(preds, axis=1)
print(classification_report(test_y, preds))
