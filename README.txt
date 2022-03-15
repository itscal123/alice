
=================================================================
ALICE: A Witty Chatbot
    + programmed by Isabel Flores, Calvin Lee, and Ethan Song
=================================================================
# Directions for running demonstration (app.py)
    1. Install all libraries using the command "pip install -r requirements.txt"
    2. Run the command "streamlit run app.py. The program will open in a new browser tab.
    3. Wait for the models to initialize.
    4. Once all loaded, type away!

# Mention the dataset your team used, with the URL for each
Datasets used:
• Sarcasm Corpus v2 (https://nlds.soe.ucsc.edu/sarcasm2)
• Cornell Movie Dialogs Corpus (https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

# Mention the external libraries your team used (libraries that are not default to python but
# need to be downloaded for your code to run), with the URL for each
Libraries used:
• matplotlib (https://matplotlib.org/)
• nltk (https://www.nltk.org/)
• numpy (https://numpy.org/)
• pandas (https://pandas.pydata.org/)
• pyparsing (https://pypi.org/project/pyparsing/)
• scikit_learn (https://scikit-learn.org/stable/)
• sentence_transformers (https://www.sbert.net/)
• streamlit (https://streamlit.io/)
• streamlit_chat (https://pypi.org/project/streamlit-chat/)
• torch (https://pytorch.org/)
• tqdm (https://tqdm.github.io/)
• transformers (https://huggingface.co/docs/transformers/index)


# List the publicly available code(s) you used in your project. Please provide the URL for the
# code and mention if you modified the code or not. If you modified the code, please mention
# the number of lines your team modified or added.
Publicly available codes used:
• SBERT Semantic Search Example (https://www.sbert.net/examples/applications/semantic-search/README.html)
    - Modified/added ~5 lines of code
• Fine-tuning BERT for sentence classification-- not used in final chatbot, but added to show attempt
    (# https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/
    #:~:text=It%20is%20designed%20to%20pre,wide%20range%20of%20NLP%20tasks.%E2%80%9D)
    - Modified/added approximately 200 lines of code
• PyTorch Chatbot Tutorial (https://pytorch.org/tutorials/beginner/chatbot_tutorial.html#preparations). 
    - Modified/added approximately 300 lines of code


# List the code(s) written entirely by your team. Please roughly mention how
# many lines of code is present in each and provide a brief description (for each) of what the
# code does.
Scripts/functions written by our team:
[User Interface]
• app.py [100]: Loads and runs the fully trained IR and Generative model.

[IR Model]
• BERT.py [275] BERT/IR Model class file. Everything needed to initialize and operate ALICE IR Model. Including
    computing embeddings.
• data_processor.py [140] data preprocessing functions + 1 data statistic function

[Gen Model]
• data.py [350]: Creates the data object and necessary data preprocessing used during training and inference.
• train.py [600]: Creates and trains the encoder-decoder model with Attention
• model.py [100]: Generative Model wrapper class. Loads fully trained model and generates responses.
