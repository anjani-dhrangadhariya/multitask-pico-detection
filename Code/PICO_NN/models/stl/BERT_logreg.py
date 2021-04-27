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

# sklearn crfsuite
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# pyTorch CRF
from torchcrf import CRF

# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig, BertForTokenClassification
from transformers import AdamW, BertConfig 
from transformers import get_linear_schedule_with_warmup

# Import data getters
from Data_builders.data_builder import get_data_loaders_, get_data_loaders, get_vocab_and_tag_maps, get_label_counts
from Data_builders.data_builder_indPIO import get_data_loaders as dl_individual

global clf_P_num_labels, clf_P_fine_num_labels
experiment_type = 'seq_lab' # another choice NER, seq_lab
set_entity  = 'outcome'

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

class BERTLogReg(nn.Module):

    def __init__(self, freeze_bert, tokenizer, class_weights):
        super(BERTLogReg, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)
        
        #Freeze bert layers: if True, the freeze BERT weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.tokenizer = tokenizer

        # log reg
        self.hidden2tag = nn.Linear(768, clf_P_fine_num_labels)

        # loss calculation
        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None):

        # BERT
        outputs = self.bert_layer(
            input_ids,
            attention_mask = attention_mask
        )

        # output 0 = batch size 6, tokens 512, each token dimension 768 [CLS] token
        # output 1 = batch size 6, each token dimension 768
        # output 2 = layers 13, batch 6 (hidden states), tokens 512, each token dimension 768
        sequence_output = outputs[2] # Last layer of each token prediction

        num_layer_sum = 4
        summed_last_4_layers = torch.stack(sequence_output[:num_layer_sum]).mean(0)

        # mask the unimportant tokens before log_reg
        mask = (
            (input_ids != self.tokenizer.pad_token_id)
            & (input_ids != self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token))
            & (labels != 100)
        )
        labels *= mask.long()

        mask_expanded = mask.unsqueeze(-1).expand(summed_last_4_layers.size())
        summed_last_4_layers *= mask_expanded.float()

        # linear
        probablity = F.relu ( self.hidden2tag( summed_last_4_layers ) )
        max_probs = torch.max(probablity, dim=2)         
        logits = max_probs.indices
        # logits = max_probs.indices.flatten()

        # calculate loss
        loss = self.loss_fct(probablity.view(-1, clf_P_fine_num_labels), labels.view(-1))

        return loss, logits, labels, mask
