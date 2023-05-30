# argument and os tools
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# utility functions
import sys
sys.path.append("../")
import utils.req_functions as rf
# tensorflow for tokenizing and loading saved model
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf

model_path = os.path.join("..","out","rnn_model_comments.keras")
saved_model = tf.keras.models.load_model(model_path)
corpus = os.path.join("..","data","corpus_data.csv")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

def make_parse():
    parser = argparse.ArgumentParser("LSTM Text Generation")
    parser.add_argument("seed_text", type=str)
    parser.add_argument("next_words", type=int, default=10)
    args = parser.parse_args()
    return args

def prompt_response(args):
    text = rf.generate_text(tokenizer = tokenizer,
                            seed_text = args.seed_text, 
                            next_words = args.next_words,
                            model = saved_model,
                            max_sequence_len = 288) # input shape
    print(text)
    return text

def main():
    arguments = make_parse()
    reply = prompt_response(arguments)
    return None

if __name__ == "__main__":
    main()