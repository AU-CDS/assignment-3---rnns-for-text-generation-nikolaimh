print("Starting train_model.py")

# utility functions
import sys
sys.path.append("../")
import utils.req_functions as rf

# data processing tools
import string
import os
import pandas as pd
import numpy as np
np.random.seed(32)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(87)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

###warnings.resetwarnings()

outpath = os.path.join("..","out") # defining outpath for results

print("Defining functions...")
# creating a data-loading function
def load_data():
    # empty list to contain comments 
    all_comments = []
    data_path = os.path.join("..","data")

    print("   (fetching comments...)")
    for filename in os.listdir(data_path): # for each file in the list of files in data_path
        if "Comments" in filename: # excluding the non-comment datasets
            comment_df = pd.read_csv(os.path.join(data_path, filename), nrows = 200) # save subset of comments in dataframe
                                                                                      # delete nrows = 100 to use all comments
            #comment_df["commentBody"].replace("", np.nan)
            comment_df = comment_df.dropna(subset=["commentBody"]) # remove missing/NaN comments
            all_comments.extend(list(comment_df["commentBody"].values)) # add each comment body to the list

    print("   (cleaning text...)")
    corpus = [rf.clean_text(x) for x in all_comments] # using clean_text function from utils
    return corpus

# creating a data-processing function
def processing(corpus): 
    tokenizer = Tokenizer()

    print("   (fitting tokenizer...)")
    tokenizer.fit_on_texts(corpus) # applying tokenizer to the input
    total_words = len(tokenizer.word_index) + 1

    print("   (getting input sequence lengths...)")
    inp_sequences = rf.get_sequence_of_tokens(tokenizer, corpus) # using util function to find sequence lengths
    
    print("   (generating padded sequences...)")
    predictors, label, max_sequence_len = rf.generate_padded_sequences(inp_sequences,total_words) # making all inputs equal length
    return predictors, label, max_sequence_len, total_words, tokenizer

# training a model with given inputs
def train_model(predictors, label, max_sequence_len, total_words, tokenizer):
    # create model
    print("   (creating model...)")
    model = rf.create_model(max_sequence_len, total_words) 

    # training model with parameters below
    print("   (training model...)")
    history = model.fit(predictors, 
                        label,
                        epochs=20, # reiterate and update the model 20 times to finish
                        batch_size=128, # testing on batch of size 128 before updating model
                        verbose=1)
    return model

# saving trained model to predefined outpath
def save_func(model_name, corpus):
    model_path = os.path.join(outpath,"rnn_model_comments.keras")
    tf.keras.models.save_model(
        model_name, model_path, overwrite=True, save_format="keras",
    )
    corpus_path = os.path.join("..","data","corpus_data.csv")
    corpus_df = pd.DataFrame(corpus)
    corpus_df.to_csv(corpus_path)
    return None

def main():
    # loading data from data folder
    print("Loading data...")
    corpus_data = load_data()
    # tokenizing data and getting predictions
    print("Processing...")
    predictions, group, length, words, tokenizer = processing(corpus_data)
    # training models for set number of epochs (currently set to 20)
    print("Training model...")
    trained_model = train_model(predictions, group, length, words, tokenizer)
    # saving trained model to be called on by prompt.py
    print("Saving model...")
    save_func(trained_model, corpus_data)
    print("Model saved to the out folder.")
    
    return None

if __name__ == "__main__":
    main()