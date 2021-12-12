# author: Jarek Brown

# this file relies heavily on huggingface's BERT functions and classes
# documentation for huggingface BERT: https://huggingface.co/docs/transformers/model_doc/bert?highlight=berttokenizer

# this is intended to extract the attention tensors from the BERT base model
# the model is a locally fine-tuned  model


from transformers import BertConfig, BertModel, BertTokenizer, BertForPreTraining
from contextlib import redirect_stdout
import torch as tr
import csv
import numpy as np
import random
import os
import chardet
import codecs
import tensorflow as tf

from transformers.file_utils import is_tf_available, is_torch_available


# path to currenty directory
PATH = os.getcwd()


# function finds the path to the specified file
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


# make sure to change the input file encoding ( use "iconv -t utf8 <filename> -o <newfilename>" on Linux)
input_file = find('train1.csv', PATH)

# this should handle the model related information
config_file = find('bert_config.json', PATH)


# read in the .tsv file (I used the same function as Google's BERT code)
with tf.io.gfile.GFile(input_file, 'r') as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    lines = []
    for line in reader:
        lines.append(line)
f.close()


# function for setting the seed for reproducing results
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        tr.manual_seed(seed)
        tr.cuda.manual_seed_all(seed)  # safe even if cuda is N/A
    if is_tf_available():
        tf.random.set_seed(seed)


# calls the seed function (change if you want)
set_seed(1)

# config object to use local pre-trained BERT model
# this will output the attentions, but can also output hidden states (change value to true)
config = BertConfig.from_pretrained(
    config_file, output_attentions=True, output_hidden_states=False, local_files_only=True)
model = BertModel(config)

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased')


# will need to change this to accept a file
for (i, l) in enumerate(lines):
    if i == 0:
        continue
    tokens = tokenizer.encode(l, return_tensors="pt")

outputs = model(tokens)


# writing out the tokens (not super helpful)
# with open('out1.txt', 'w') as f:
#     with redirect_stdout(f):
#         print(tokens)

# writing out the attentions to 'attentions-output.txt'
with open('attentions-output.txt', 'w') as f:
    with redirect_stdout(f):
        print(outputs)
