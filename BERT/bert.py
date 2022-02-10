import torch
from transformers import BertTokenizer, BertModel
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt


# The following initializations are just so I could test out my embed_message function.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )


# Do we need this? 
'''
# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
'''


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
    #print (tokenized_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Display the words with their indeces.
    # for tup in zip(tokenized_text, indexed_tokens):
    #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    # Create segment IDs: assign each word in the first sentence plus the ‘[SEP]’ token a 0, 
    # and all tokens of the second sentence a 1.
    segments_ids = [1] * len(tokenized_text)

    #print (segments_ids)

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
    token_embeddings = token_embeddings.permute(1,0,2)

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    #print ("Our final sentence embedding vector of shape:", sentence_embedding.size())
    
    return sentence_embedding

# This is me testing the output of my function. Pretty sure it works.
print(embed_message("hello?", tokenizer, model))