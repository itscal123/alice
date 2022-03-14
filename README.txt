
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
• YYY (URL2)
• PyTorch Chatbot Tutorial (https://pytorch.org/tutorials/beginner/chatbot_tutorial.html#preparations). 
    - Modified/added approximately 300 lines of code


# List the code(s) written entirely by your team. Please roughly mention how
# many lines of code is present in each and provide a brief description (for each) of what the
# code does.
Scripts/functions written by our team:
[User Interface]
• app.py: Loads and runs the fully trained IR and Generative model.

[IR Model]
• BERT.py BERT/IR Model class file. Everything needed to initialize and operate ALICE IR Model.
• data_processor.py data preprocessing functions + 1 data statistic function

[Gen Model]
• data.py: Creates the data object and necessary data preprocessing used during training and inference.
• train.py: Creates and trains the encoder-decoder model with Attention
• model.py: Generative Model wrapper class. Loads fully trained model and generates responses. 
