##################################################################################
# Imports
##################################################################################
# staple imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import glob
import numpy as np
import pandas as pd
import time
import datetime
import argparse
import pdb
import json
import random

# numpy essentials
from numpy import asarray
import numpy as np

# pyTorch essentials
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
from torch import LongTensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# keras essentials
from keras.preprocessing.sequence import pad_sequences

# sklearn
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix

# pyTorch CRF
from torchcrf import CRF
from models.mtl.ConditionalRandomFields import CRF as ConditionalRandomFields

# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig, BertForTokenClassification
from transformers import AdamW, BertConfig 
from transformers import get_linear_schedule_with_warmup
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config

# Import data getters
from Data_builders.data_builder import get_data_loaders_, get_data_loaders, get_vocab_and_tag_maps, get_label_counts
from Data_builders.data_builder_indPIO import get_data_loaders as dl_individual
from models.HelperFunctions import get_packed_padded_output

##################################################################################
# global variables XXX Set your pretrained model name here IMPORTANT
##################################################################################
global configurations_dictionary, set_phase, saved_models, clf_P_num_labels, clf_P_fine_num_labels, pretrained_model, experiment_type, fineGrained
configurations_dictionary = dict()
"""
Entity settings (var: set_entity) requires choosing betweeen "participant", "intervention" and "outcome"

Experiment type (var: experiment_type) requires choosing between "ner", "bioes" and "seq_lab" depending upon how you want to encode the labels

Choice of the model architecture (var: pretrained_model) requires choosing between the architectures.
"""

fineGrained = True
set_entity  = 'outcome'
experiment_type = 'seq_lab' # another choice NER, seq_lab

if experiment_type == 'seq_lab':
    clf_P_num_labels = 2
if experiment_type == 'ner':
    clf_P_num_labels = 3
if experiment_type == 'bioes':
    clf_P_num_labels = 5

if set_entity ==  'participant':
    clf_P_fine_num_labels = 5
elif set_entity ==  'intervention':
    clf_P_fine_num_labels = 8
elif set_entity ==  'outcome':
    clf_P_fine_num_labels = 7

class GPT2LSTMattenLogRegCRF(nn.Module):

    def __init__(self, freeze_bert, tokenizer, device, bidirectional, class_weight_c = None, class_weight_f = None):
        super(GPT2LSTMattenLogRegCRF, self).__init__()

        self.hidden_dim = 512
        self.tokenizer = tokenizer
        self.device = device
        self.bidirectional = bidirectional
        self.class_weight_c = class_weight_c
        self.class_weight_f = class_weight_f

        #Instantiating BERT model object 
        self.gpt2_layer = GPT2Model.from_pretrained('gpt2', output_hidden_states=True, output_attentions=False)
        
        #Freeze bert layers: if True, the freeze BERT weights
        if freeze_bert:
            for p in self.gpt2_layer.parameters():
                p.requires_grad = False

        # lstm layer
        self.lstm_layer = nn.LSTM(input_size=768, hidden_size = 512, num_layers = 1, bidirectional=bidirectional, batch_first=True)

        # attention mechanism
        if bidirectional == True:
            self.self_attention = nn.MultiheadAttention(self.hidden_dim * 2, 1, bias=True) # attention mechanism from PyTorch
        else:
            self.self_attention = nn.MultiheadAttention(self.hidden_dim, 1, bias=True)           

        # log reg
        if bidirectional == True:
            self.hidden2tag = nn.Linear(1024, clf_P_num_labels)
            self.hidden2tag_fine = nn.Linear(1024, clf_P_fine_num_labels)
        else:
            self.hidden2tag = nn.Linear(512, clf_P_num_labels)
            self.hidden2tag_fine = nn.Linear(512, clf_P_fine_num_labels)

        # crf (coarse)
        self.crf_layer = CRF(clf_P_num_labels, batch_first=True)
        # crf (fine)
        self.crf_layer_fine = CRF(clf_P_fine_num_labels, batch_first=True)

    def forward(self, args, input_ids=None, attention_mask=None, P_labels=None, P_f_labels=None):

        # BERT
        outputs = self.gpt2_layer(
            input_ids,
            attention_mask = attention_mask
        )

        # output 0 = batch size 6, tokens 512, each token dimension 768 [CLS] token
        # output 1 = batch size 6, each token dimension 768
        # output 2 = layers 13, batch 6 (hidden states), tokens 512, each token dimension 768
        sequence_output = outputs[2] # Last layer of each token prediction

        num_layer_sum = 4
        summed_last_4_layers = torch.stack(sequence_output[:num_layer_sum]).mean(0)

        # lstm with masks (same as attention masks)
        packed_input, perm_idx, seq_lengths = get_packed_padded_output(summed_last_4_layers, input_ids, attention_mask, self.tokenizer)
        packed_output, (ht, ct) = self.lstm_layer(packed_input)

        # Unpack and reorder the output
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = output[unperm_idx] # lstm_output.shape = shorter than the padded torch.Size([6, 388, 512])
        seq_lengths_ordered = seq_lengths[unperm_idx]

        # shorten the labels as per the batchsize
        P_labels = P_labels[:, :lstm_output.shape[1]]
        P_f_labels = P_f_labels[:, :lstm_output.shape[1]]

        # Apply mask before calculating the matmul of inputs and the K, Q, V weights
        attention_mask_ = attention_mask[:, :lstm_output.shape[1]]
        attention_mask_ = attention_mask_.bool()

        # apply more weight

        # Attention should be applied to the LSTM outputs here
        # attention_applied, attention_weights = self.attn(lstm_output, attention_mask.float())
        # attention_applied, attention_weights = self.self_attention( lstm_output, lstm_output, lstm_output, key_padding_mask=attention_mask_.permute(1, 0), need_weights=True, attn_mask=None )
        attention_applied, attention_weights = self.self_attention( lstm_output, lstm_output, lstm_output, key_padding_mask=None, need_weights=True, attn_mask=None )

        # mask the unimportant tokens after attention is applied
        mask = (
            #(input_ids[:, :lstm_output.shape[1]] != self.tokenizer.unk_token_id)
            (input_ids[:, :lstm_output.shape[1]] != 50256)
            # & (input_ids[:, :lstm_output.shape[1]] != self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token))
            & (input_ids[:, :lstm_output.shape[1]] != 50256)
            & (P_labels != 100)
        )

        # on the first time steps
        for eachIndex in range( mask.shape[0] ):
            mask[eachIndex, 0] = True

        mask_expanded = mask.unsqueeze(-1).expand(attention_applied.size())
        attention_applied *= mask_expanded.float()
        P_labels *= mask.long()
        P_f_labels *= mask.long()

        # log reg (coarse)
        probablity = F.relu ( self.hidden2tag( attention_applied ) )
        # log reg (fine)
        probablity_fine = F.relu ( self.hidden2tag_fine( attention_applied ) )

        # CRF emissions (coarse)
        #loss_coarse = self.crf_layer(probablity, reduction='token_mean')
        loss_coarse = self.crf_layer(probablity, P_labels, mask=mask, reduction='token_mean')

        emissions_coarse = self.crf_layer.decode( probablity )
        emissions_c = [item for sublist in emissions_coarse for item in sublist] # flatten the nest list of emissions

        # CRF emissions (fine)
        #loss_fine = self.crf_layer_fine(probablity_fine, reduction='token_mean')
        loss_fine = self.crf_layer_fine(probablity_fine, P_f_labels, mask=mask, reduction='token_mean')

        emissions_fine = self.crf_layer_fine.decode( probablity_fine )
        emissions_f = [item for sublist in emissions_fine for item in sublist] # flatten the nest list of emissions

        loss = abs( loss_coarse ) + abs( loss_fine )

        return loss_coarse,  torch.Tensor(emissions_coarse), P_labels, loss_fine,  torch.Tensor(emissions_fine), P_f_labels, mask, loss
