##################################################################################
# Imports
##################################################################################
import os
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import argparse
import pdb
import glob
import random 

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim


# keras essentials
from keras.preprocessing.sequence import pad_sequences

# sklearn
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.model_selection import KFold

# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import AdamW, BertConfig 
from transformers import get_linear_schedule_with_warmup

# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import AdamW 
from transformers import get_linear_schedule_with_warmup

##################################################################################
# Set all the seed values
##################################################################################
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def last_index(myList):
    return len(myList)-1

def get_vocab_and_tag_maps(code):

    ##################################################################################
    # Import word vocabulary
    ##################################################################################
    words_path = '/Code/PICO_NN/req_files/words.txt'
    vocab = {}
    with open(words_path) as f:
        for i, l in enumerate(f.read().splitlines()):
            vocab[l] = i

    # Import tags vocabulary 
    if code == 'P':
        tags_path = '/Code/PICO_NN/req_files/tagsP.txt'
    if code == 'I':
        tags_path = '/Code/PICO_NN/req_files/tagsI.txt'
    if code == 'O':
        tags_path = '/Code/PICO_NN/req_files/tagsO.txt'

    tag_map = dict()
    with open(tags_path) as f:
        for i, l in enumerate(f.read().splitlines()):
            tag_map[l] = i
    
    return vocab, tag_map

def readFineGrainedLabels(train_label_files, train_label_dir):
    labels = dict()

    for each_file in train_label_files:
        with open(os.path.join(train_label_dir, each_file), 'r') as f:
            sentence_labels_encoded = []
            sentence_labels_encoded = [0 if token=='O' else int(token) for token in f.read().splitlines() ]
            labels[each_file.split('.')[0]] = sentence_labels_encoded
    train_label_files_ = [each_file.split('.')[0] for each_file in train_label_files]
    return labels, train_label_files_

def readLabels(train_label_files, train_label_dir, tag_map):

    labels = dict()

    for each_file in train_label_files:
        with open(os.path.join(train_label_dir, each_file), 'r') as f:
            sentence_labels_encoded = []
            for token in f.read().splitlines():
                if token == '00':
                    token = '0'
                sentence_labels_encoded.append(tag_map[token])
            labels[each_file.split('.')[0]] = sentence_labels_encoded
    train_label_files_ = [each_file.split('.')[0] for each_file in train_label_files]
    return labels, train_label_files_

def readBIOESLabels(train_label_files, train_label_dir, tag_map):

    labels = dict()

    for each_file in train_label_files:

        with open(os.path.join(train_label_dir, each_file), 'r') as f:
            sentence_labels_encoded = []
            for token in f.read().splitlines():
                if token == '00':
                    token = '0'
                sentence_labels_encoded.append(tag_map[token])

            # Convert normal labels to BIOES labels
            bioes_labels = []
            for i, eachLabel in enumerate(sentence_labels_encoded):
                if eachLabel == 0:
                    bioes_labels.append(0)
                if eachLabel == 1:
                    if i == 0: # first index
                        if sentence_labels_encoded[i] == 1 and sentence_labels_encoded[i+1] == 1:
                            bioes_labels.append(1)
                        elif sentence_labels_encoded[i] == 1 and sentence_labels_encoded[i+1] != 1:
                            bioes_labels.append(4)
                    
                    if i == len(sentence_labels_encoded)-1: # last index
                        if sentence_labels_encoded[i-1] != 1:
                            bioes_labels.append(4)
                        if sentence_labels_encoded[i-1] == 1:
                            bioes_labels.append(2)

                    else:
                        if sentence_labels_encoded[i-1] != 1 and sentence_labels_encoded[i+1] != 1:
                            bioes_labels.append(4)
                        if sentence_labels_encoded[i-1] != 1 and sentence_labels_encoded[i+1] == 1:
                            bioes_labels.append(1)
                        if sentence_labels_encoded[i-1] == 1 and sentence_labels_encoded[i+1] == 1:
                            bioes_labels.append(2)
                        if sentence_labels_encoded[i-1] == 1 and sentence_labels_encoded[i+1] != 1:
                            bioes_labels.append(3)

            labels[each_file.split('.')[0]] = bioes_labels # Each filename is assigned the token labels here
    train_label_files_ = [each_file.split('.')[0] for each_file in train_label_files]
    return labels, train_label_files_

def getLabels(tag_map, code, experiment_type, exp_args):

    if code == 'P':
        train_label_dir = '/ebm_nlp_2_00/annotations/aggregated/starting_spans/participants/annot/train'
        test_label_dir = '/ebm_nlp_2_00/annotations/aggregated/starting_spans/participants/annot/test/gold'
        test_label_dir_incorpus = '/Data/PICO_ann_coarsegrained/labels/participants/annot'

        train_label_fine_dir = '/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/participants/train/'
        test_label_fine_dir = '/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/participants/test/gold/'
        test_label_fine_dir_incorpus = '/Data/PICO_ann_finegrained/labels/participant/'

    elif code == 'I':
        train_label_dir = '/ebm_nlp_2_00/annotations/aggregated/starting_spans/interventions/train'
        test_label_dir = '/ebm_nlp_2_00/annotations/aggregated/starting_spans/interventions/test/gold'
        test_label_dir_incorpus = '/Data/PICO_ann_coarsegrained/labels/intervention/annot'

        train_label_fine_dir = '/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/interventions/train/'
        test_label_fine_dir = '/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/interventions/test/gold/'
        test_label_fine_dir_incorpus = '/Data/PICO_ann_finegrained/labels/labels/intervention/'

    elif code == 'O':
        train_label_dir = '/ebm_nlp_2_00/annotations/aggregated/starting_spans/outcomes/train'
        test_label_dir = '/ebm_nlp_2_00/annotations/aggregated/starting_spans/outcomes/test/gold'
        test_label_dir_incorpus = '/Data/PICO_ann_coarsegrained/labels/outcome/annot'

        train_label_fine_dir = '/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/outcomes/train/'
        test_label_fine_dir = '/ebm_nlp_2_00/annotations/aggregated/hierarchical_labels/outcomes/test/gold/'
        test_label_fine_dir_incorpus = '/Data/PICO_ann_finegrained/labels/outcome/'

    else:
        raise "Provide one of these codes (P, I, O)" 
    
    train_label_files = os.listdir(train_label_dir)
    if experiment_type == 'bioes':
        train_encoded_labels, train_files = readBIOESLabels(train_label_files, train_label_dir, tag_map)
    elif experiment_type == 'seq_lab':
        train_encoded_labels, train_files = readLabels(train_label_files, train_label_dir, tag_map)
        train_encoded_fine_labels, train_files = readFineGrainedLabels(train_label_files, train_label_fine_dir)


    test_label_files = os.listdir(test_label_dir)
    if experiment_type == 'bioes':
        test_encoded_labels, test_files = readBIOESLabels(test_label_files, test_label_dir, tag_map)
    elif experiment_type == 'seq_lab':
        test_encoded_labels, test_files = readLabels(test_label_files, test_label_dir, tag_map)
        test_encoded_fine_labels, test_files = readFineGrainedLabels(test_label_files, test_label_fine_dir)


    test_label_files_incorpus = os.listdir(test_label_dir_incorpus)
    if experiment_type == 'bioes':
        test_encoded_labels_incorpus, test_files_incorpus = readBIOESLabels(test_label_files_incorpus, test_label_dir_incorpus, tag_map)
    elif experiment_type == 'seq_lab':
        test_encoded_labels_incorpus, test_files_incorpus = readLabels(test_label_files_incorpus, test_label_dir_incorpus, tag_map)
        test_encoded_fine_labels_incorpus, test_files_incorpus = readFineGrainedLabels(test_label_files_incorpus, test_label_fine_dir_incorpus)


    if exp_args.fineGrained == True:
        return train_encoded_labels, test_encoded_labels, test_encoded_labels_incorpus, train_encoded_fine_labels, test_encoded_fine_labels, test_encoded_fine_labels_incorpus, train_files
    else:
        return train_encoded_labels, test_encoded_labels, test_encoded_labels_incorpus, train_files

# the function truncatedlist of lists to a max length and add special tokens
def truncateSentence(sentence, trim_len):
    """
    Truncates the sequence length to (MAX_LEN - 2). 
    Negating 2 for the special tokens
    """
    trimmedSentence = []
    if  len(sentence) > trim_len:
        trimmedSentence = sentence[:trim_len]
    else:
        trimmedSentence = sentence

    assert len(trimmedSentence) <= trim_len
    return trimmedSentence

# add the special tokens in the end of sequences
def addSpecialtokens(eachText, start_token, end_token):
    insert_at_start = 0
    eachText[insert_at_start:insert_at_start] = [start_token]

    insert_at_end = len(eachText)
    eachText[insert_at_end:insert_at_end] = [end_token]

    assert eachText[0] == start_token
    assert eachText[-1] == end_token

    return eachText

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer, max_length, pretrained_model):

    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """
    dummy_label = 100 # Could be any kind of labels that you can mask
    tokenized_sentence = []
    labels = []
    labels_fg = []
    printIt = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.encode(word, add_special_tokens = False)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        if n_subwords == 1:
            labels.extend([label] * n_subwords)
            labels_fg.extend([label] * n_subwords)
        else:
            labels.extend([label])
            labels.extend([dummy_label] * (n_subwords-1))

    assert len(tokenized_sentence) == len(labels)

    # Truncate the sequences (sentence and label) to (max_length - 2)
    truncated_sentence = truncateSentence(tokenized_sentence, (max_length - 2))
    truncated_labels = truncateSentence(labels, (max_length - 2))
    assert len(truncated_sentence) == len(truncated_labels)

    # Add special tokens CLS and SEP
    if 'bert' in pretrained_model.lower():
        speTok_sentence = addSpecialtokens(truncated_sentence, tokenizer.cls_token_id, tokenizer.sep_token_id)
    elif 'gpt2' in pretrained_model.lower():
        speTok_sentence = addSpecialtokens(truncated_sentence, tokenizer.bos_token_id, tokenizer.eos_token_id)
    speTok_labels = addSpecialtokens(truncated_labels, 0, 0)

    # PAD the sequences to max length
    if 'bert' in pretrained_model.lower():
        input_ids = pad_sequences([ speTok_sentence ] , maxlen=max_length, value=tokenizer.pad_token_id, padding="post")
    elif 'gpt2' in pretrained_model.lower():
        input_ids = pad_sequences([ speTok_sentence ] , maxlen=max_length, value=tokenizer.unk_token_id, padding="post")

    input_labels = pad_sequences([ speTok_labels ] , maxlen=max_length, value=0, padding="post")

    assert len(input_ids.squeeze()) == max_length
    assert len(input_labels.squeeze()) == max_length

    return input_ids.squeeze(), input_labels.squeeze()

def get_train_dev_sets(complete_dataset):
    total_length = len(complete_dataset)
    print('total length of the full dataset: ', total_length)
    eighty_total = (total_length * 90) / 100
    twenty_total = (total_length * 10) / 100

    train_data = complete_dataset[:round(eighty_total)]
    dev_data = complete_dataset[round(eighty_total):]

    print('Length of train dataset: ', len(train_data))
    print('Length of development dataset: ', len(dev_data))

    return train_data, dev_data

def normalize_weights(weights):
    '''
    Normalization by scaling: https://docs.tibco.com/pub/spotfire/7.0.0/doc/html/norm/norm_scale_between_0_and_1.htm
    '''
    eps = 1e-8
    min_val = min(weights)
    max_val = max(weights)

    weights_normalized = [(eachWeightTerm - min_val) / (max_val - min_val) for eachWeightTerm in weights]
    weights_normalized = np.asarray(weights_normalized, dtype=np.float32) # normalize weights between 0 and 1
    # print('normalize weights between 0 and 1: ', weights_normalized)
    
    weights_normalized = weights_normalized + eps
    # print('Epsilon added: ', weights_normalized)

    #weights_normalized = weights_normalized / 4.0
    # print('Divided by 10: ', weights_normalized)

    weights_normalized = weights_normalized + 1
    # print('Unit added to 1: ', weights_normalized)

    print('Final weights for CRF layer: ', weights_normalized)

    return weights_normalized

def get_data_loaders(batch_size, MAX_LEN_new, tokenizer, pretrained_model, experiment_type, exp_args):

    vocab, tag_map = get_vocab_and_tag_maps('I')

    # Get the span labels
    # Get the fine grained labels
    print('Fetching fine grained labels')
    train_labels, test_labels, test2_labels, train_fine_labels, test_fine_labels, test_fine2_labels, train_label_files_  = getLabels(tag_map, 'I', experiment_type, exp_args)

    train_list = list(train_labels.values())
    train_fine_list = list(train_fine_labels.values())

    flat_train_list = [item for sublist in train_list for item in sublist]
    flat_train_fine_list = [item for sublist in train_fine_list for item in sublist]

    # Compute class weights for coarse grained labels
    class_weights_ = compute_class_weight('balanced',
                                        np.unique(flat_train_list),
                                        flat_train_list)

    # Compute class weights for coarse grained labels
    class_weights_fine_ = compute_class_weight('balanced',
                                        np.unique(flat_train_fine_list),
                                        flat_train_fine_list)
                                        

    # Normalize classweights to unit
    class_weights = normalize_weights(class_weights_) # Weights for coarse grained according to class label freq
    # class_weights_coarse = []
    # for i, classWeights in enumerate(class_weights):
    #     if i == 0:
    #         class_weights_coarse.append(1.0)
    #     else:
    #         class_weights_coarse.append(1.02)
    # class_weights = np.asarray(class_weights_coarse, dtype=np.float32) # Unit weight for coarse-grained task


    class_weights_fine = normalize_weights(class_weights_fine_)
    # class_weights_fine = []
    # for i, classWeights in enumerate(class_weights_fine_):
    #     if i == 0:
    #         class_weights_fine.append(1.0)
    #     else:
    #         class_weights_fine.append(1.02)
    # class_weights_fine = np.asarray(class_weights_fine, dtype=np.float32) # Unit weight for coarse-grained task

    # Get the data (titles and abstracts from training and test set) tokens 
    train_abstracts_encoded = dict()
    test_abstracts_encoded = dict()
    data_dir = '/ebm_nlp_2_00/documents/'

    data_branches = [ name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name)) ]

    for each_dir in data_branches:
        files = os.listdir(os.path.join(data_dir, each_dir))
        for each_file in files:
            if each_file.endswith('.tokens'):
                with open(os.path.join(data_dir, each_dir, each_file), 'r') as f:
                    sentence = []
                    sentence = [ token for token in f.read().splitlines() ]

                    if 'train' in each_dir and each_file.split('.')[0] in train_label_files_:
                        train_abstracts_encoded[each_file.split('.')[0]] = sentence
                    elif 'test' in each_dir :
                        test_abstracts_encoded[each_file.split('.')[0]] = sentence
    print('Loading the abstracts and the abstract labels completed...')

    test2_abstracts_encoded = dict()
    incorpus_dir = '/Data/PICO_ann_finegrained/tokens'
    incorpus_files = os.listdir(incorpus_dir)
    for each_file in incorpus_files:
        with open(os.path.join(incorpus_dir, each_file), 'r') as f:
            sentence = []
            sentence = [ token for token in f.read().splitlines() ]           
            test2_abstracts_encoded[each_file.split('.')[0]] = sentence

    assert len(test2_abstracts_encoded) == len(test2_labels)
    print('Loading test tokens from in-house corpus completed...')
    
    train_keys = list(train_abstracts_encoded.keys())
    # train_keys = train_keys[0:10]


    test_keys = list(test_abstracts_encoded.keys())
    test2_keys = list(test2_abstracts_encoded.keys())

    # XXX Training set: tokenize, preserve labels, truncate, add special tokens and pad to the MAX_LEN_new
    lol = 0
    tokenized_train = []
    tokenized_train_fine = []
    for eachKey in train_keys:
        if eachKey in train_abstracts_encoded and eachKey in train_labels:
            temp = tokenize_and_preserve_labels(train_abstracts_encoded[eachKey], train_labels[eachKey], tokenizer, MAX_LEN_new, pretrained_model)
            tokenized_train.append( temp )
            if exp_args.fineGrained == True:
                temp_fine = tokenize_and_preserve_labels(train_abstracts_encoded[eachKey], train_fine_labels[eachKey], tokenizer, MAX_LEN_new, pretrained_model)
                tokenized_train_fine.append(temp_fine)


    print('Retrieved training fine label set of size: ', len(tokenized_train_fine))      

    # XXX Test set:  tokenize, preserve labels, truncate, add special tokens and pad to the MAX_LEN_new
    lol = 0
    tokenized_test = []
    tokenized_test_fine = []
    for eachKey in test_keys:
        if eachKey in test_abstracts_encoded and eachKey in test_labels:
            temp = tokenize_and_preserve_labels(test_abstracts_encoded[eachKey], test_labels[eachKey], tokenizer, MAX_LEN_new, pretrained_model)
            tokenized_test.append( temp )
            if exp_args.fineGrained == True:
                temp_fine = tokenize_and_preserve_labels(test_abstracts_encoded[eachKey], test_fine_labels[eachKey], tokenizer, MAX_LEN_new, pretrained_model)
                tokenized_test_fine.append(temp_fine)

    print('Retrieved test (EBM-NLP) fine label set of size: ', len(tokenized_test_fine))   

    lol = 0
    tokenized_test2 = []
    tokenized_test2_fine = []
    for eachKey in test2_keys:
        if eachKey in test2_abstracts_encoded and eachKey in test2_labels:
            temp = tokenize_and_preserve_labels(test2_abstracts_encoded[eachKey], test2_labels[eachKey], tokenizer, MAX_LEN_new, pretrained_model)
            tokenized_test2.append( temp )
            if exp_args.fineGrained == True:
                temp_fine = tokenize_and_preserve_labels(test2_abstracts_encoded[eachKey], test_fine2_labels[eachKey], tokenizer, MAX_LEN_new, pretrained_model)
                tokenized_test2_fine.append(temp_fine)
            lol = lol + 1

    print('Retrieved test (in corpus) fine label set of size: ', len(tokenized_test2_fine))

    print('Subword tokenization and label extension completed...')

    return train_keys, tokenized_train, tokenized_train_fine, tokenized_test, tokenized_test_fine, tokenized_test2, tokenized_test2_fine, class_weights, class_weights_fine
