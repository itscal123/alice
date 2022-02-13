import torch
import pandas as pd
from transformers import BertTokenizer, BertModel, AdamW, BertConfig
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
# logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt

print("## Loading Tokenizer and Pretrained Model ##")
# The following initializations are just so I could test out my embed_message function.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True,  # Whether the model returns all hidden-states.
                                  )

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

#### Pre-processing data ####
# based on code from: http://mccormickml.com/2019/07/22/BERT-fine-tuning/#21-download--extract

print("## Preprocessing Data ##")
# load dataset into a dataframe
# using just cornell set for simplicity -- will add the rest later
file = 'data/cornell movie-dialogs corpus/movie_lines.txt'

df = pd.read_csv(file, index_col=False,
                 sep=r'\+{3}\$\+{3}',
                 engine='python',
                 header=None,
                 skipinitialspace=True,
                 encoding='unicode_escape',
                 na_filter=False,
                 names=['lineID', 'charID', 'movieID', 'charName', 'text'])

print('Num sentences: {:,}\n'.format(df.shape[0]))
print(df.sample(10))

lines = df.text.values
# print('First 5 lines: {}'.format(lines[:5]))

max_length = 512
input_ids = []          # id mappings
attention_masks = []    # padding differentiators
token_type_ids = []     # segment ids

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

#### Training ####

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
This is basically just the code from https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
Debugging print statements have been commented out and the "teaching/demonstration code" has been omitted.
I'm thinking of using this function in our "main loop," where we continually prompt the user for a chat
message. 

This function takes in a String message, a pre-trained model tokenizer, and a BERT model. It returns a 
single sentence embedding (a vector) that corresponds to the given message. I'm pretty sure we can just 
"Plug-and-Play" a fine-tuned BERT model into here.

We're going to run dot-product or cosine-similarity on the embedding of the user's message
and the embedding of every sentence in our corpus/database/dataset/data(...?)
'''


# NOTE: I think this function can be modified to get embeddings for every sentence in a corpus... Pretty easily too...
def embed_message(message, tokenizer, model):
    # Add required start and end tokens
    marked_text = "[CLS] " + message + " [SEP]"

    # Tokenize message with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Print out the tokens.
    # print (tokenized_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Display the words with their indeces.
    # for tup in zip(tokenized_text, indexed_tokens):
    #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    # Create segment IDs: assign each word in the first sentence plus the ‘[SEP]’ token a 0, 
    # and all tokens of the second sentence a 1.
    segments_ids = [1] * len(tokenized_text)

    # print (segments_ids)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1, 0, 2)

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    # print ("Our final sentence embedding vector of shape:", sentence_embedding.size())

    return sentence_embedding


# This is me testing the output of my function. Pretty sure it works.
print(embed_message("hello?", tokenizer, model))
