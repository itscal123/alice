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
def embed_message(messages, tokenizer, model):
    # Add required start and end tokens
    # messages = np.array(messages)
    # marker = lambda x: "[CLS] " + x + " [SEP]"
    # messages = np.array(marker(message) for message in messages)
    # marked_text = "[CLS] " + messages + " [SEP]"

    # Tokenize message with the BERT tokenizer.
    # tokenized_text = tokenizer.tokenize(marked_text)

    # Print out the tokens.
    # print (tokenized_text)

    # Map the token strings to their vocabulary indeces.
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Display the words with their indeces.
    # for tup in zip(tokenized_text, indexed_tokens):
    #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    # Create segment IDs: assign each word in the first sentence plus the ‘[SEP]’ token a 0,
    # and all tokens of the second sentence a 1.
    # segments_ids = [1] * len(tokenized_text)

    # print (segments_ids)

    # this is how long our encoded vectors will be, 512 is standard, but technically not needed for our purposes
    # (we won't be returning responses > 512 words)
    # so if we needed to do any speed optimization, we could play around with this value
    max_length = 512

    input_ids = []  # id mappings
    attention_masks = []  # padding differentiators
    token_type_ids = []  # segment ids

    for msg in messages:
        # We can use the encode_plus() function to bypass having to convert ids and segment ids manually
        encoded_dict = tokenizer.encode_plus(msg,
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
    tokens_tensor = torch.cat(input_ids, dim=0)
    segments_tensor = torch.cat(token_type_ids, dim=0)

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.

    # don't need to calculate gradients
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensor)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]

    # i don't think we need token embeddings

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    # token_embeddings = torch.stack(hidden_states, dim=0)

    # Remove dimension 1, the "batches".
    # token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # Swap dimensions 0 and 1.
    # token_embeddings = token_embeddings.permute(1, 0, 2)

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]

    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    # add embedding to our embedding corpora

    # print ("Our final sentence embedding vector of shape:", sentence_embedding.size())

    return sentence_embedding