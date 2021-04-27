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
from sklearn.model_selection import KFold, StratifiedKFold

# pyTorch CRF
from torchcrf import CRF

# Transformers 
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaConfig, RobertaModel
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import AdamW 
from transformers import get_linear_schedule_with_warmup

# Import data getters
#from Data_builders.data_builder import get_data_loaders_, get_data_loaders, get_vocab_and_tag_maps, get_label_counts
from Data_builders.data_builder_indPIO import get_data_loaders as dl_individual

# import models for coarse grained and fine grained experiments
from models.stl.BERT_logreg import BERTLogReg
from models.stl.bert_lstm_logreg import BERTLSTMLogReg
from models.stl.BERT_logreg_CRF import BERTLogRegCRF
from models.stl.BERT_lstm_logreg_CRF import BERTLSTMLogRegCRF
from models.stl.BERT_lstm_attn_logreg_CRF import BERTLSTMattenLogRegCRF
from models.stl.BERT_lstm_Mattn_logreg_CRF import BERTLSTMMulattenLogRegCRF
from models.stl.BERT_lstm_attn_logreg_wCRF import BERTLSTMattenLogRegWCRF
from models.stl.BERT_lstm_Mattn_flogreg import BERTLSTMMulattenLogRegCRF as BERTLSTMMulattenfLogReg
from models.stl.GPT2_lstm_logreg_CRF import GPT2LSTMLogRegCRF
from models.stl.GPT2_lstm_attn_logreg_CRF import GPT2LSTMattenLogRegCRF


# visualization
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

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
set_entity  = 'outcome' # set the name of the entity 
pretrained_model = 'bert' # set your pretrained model here (alternative gpt2)
experiment_type = 'seq_lab' # another choice NER, seq_lab (alternative ner, bioes)

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

saved_models = []

experiment_arguments = argparse.ArgumentParser()
experiment_arguments.add_argument('-fineGrained', type=bool , default=True)
exp_args = experiment_arguments.parse_args()


##################################################################################
# set up the GPU
##################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print('Number of GPUs identified: ', n_gpu)
print(torch.cuda.get_device_name(0))
print('You are using ', torch.cuda.get_device_name(0), ' : ', device , ' device')

##################################################################################
# set all the seed values
##################################################################################
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

##################################################################################
# Helper functions
##################################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def createAttnMask(input_ids):
    # Add attention masks
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    return attention_masks

##################################################################################
# Train and evaluate 
##################################################################################

def evaluate_i(defModel, tokenizer, opti, schedulr, val_loader, args, device, epoch_number = None, filewrite=None):
    defModel.eval()
    
    mean_acc = 0
    mean_loss = 0
    count = 0
    total_val_loss_coarse = 0
    total_val_loss_fine = 0

    with torch.no_grad() :

        # collect all the evaluation predictions and ground truth here
        all_predictions_fine = []
        all_GT_fine = []

        for e_input_ids_, e_input_mask, e_labels, e_f_input_ids, e_f_input_mask, e_f_labels in val_loader:
            
            e_input_ids_ = e_input_ids_.cuda()

            with torch.cuda.device_of(e_input_ids_.data):
                e_input_ids = e_input_ids_.clone()

            # check if the model is GPT2-based
            if 'gpt2' in args.model_name:
                e_input_ids[e_input_ids_==0] = tokenizer.unk_token_id
                e_input_ids[e_input_ids_==101] = tokenizer.bos_token_id
                e_input_ids[e_input_ids_==102] = tokenizer.eos_token_id
            
            # fine grained entity labels
            e_f_input_mask = e_f_input_mask.cuda()
            e_f_labels = e_f_labels.cuda()

            # Input to the selected model
            e_loss, e_f_output, e_f_labels, mask  = defModel(e_input_ids, attention_mask=e_f_input_mask, labels=e_f_labels)
           
            mean_loss += e_loss.item()

            # write to the file here...
            for i in range(0, e_f_labels.shape[0]):
                selected_f_preds = torch.masked_select(e_f_output[i, ].cuda(), mask[i, ])
                selected_f_labs = torch.masked_select(e_f_labels[i, ].cuda(), mask[i, ])

                e_cr_fine = classification_report(y_pred=selected_f_preds.to("cpu").numpy(), y_true=selected_f_labs.to("cpu").numpy(), labels=list(range(clf_P_fine_num_labels)) , output_dict=True)
                
                if filewrite is not None:
                    with open(filewrite, 'a+') as f1_fine_file:
                        json.dump(e_cr_fine, f1_fine_file)
                        f1_fine_file.write('\n')

                all_predictions_fine.extend(selected_f_preds.to("cpu").numpy())
                all_GT_fine.extend(selected_f_labs.to("cpu").numpy())

            count += 1

        avg_val_loss_fine = mean_loss / len(val_loader)     
        writer.add_scalar('loss-validation-fine', avg_val_loss_fine, epoch_number)

        # Final classification report and confusion matrix for each epoch
        all_pred_flat_fine = np.asarray(all_predictions_fine).flatten()
        all_GT_flat_fine = np.asarray(all_GT_fine).flatten()   
        val_cr = classification_report(y_pred=all_pred_flat_fine, y_true=all_GT_flat_fine, labels=list(range(clf_P_fine_num_labels)), output_dict=True)
        print(val_cr)

        if filewrite is not None:
            val_cm = confusion_matrix(all_pred_flat_fine, all_GT_flat_fine, labels=list(range(clf_P_fine_num_labels)))
            plt.switch_backend('agg')
            sn.heatmap(val_cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='g') # font size 
            if 'ebm' in filewrite:
                evalSet = 'ebm'    
                plt.savefig('_' + set_entity + '/' + evalSet + '/' + str(args.fold) + '/' + args.model_name + "_" + "ebm.png", dpi=400)
            if 'inhouse' in filewrite:
                evalSet = 'inhouse'    
                plt.savefig('_' + set_entity + '/' + evalSet + '/' + str(args.fold) + '/' + args.model_name + "_" + "inhouse.png", dpi=400)
        
    return mean_loss / count, val_cr, all_pred_flat_fine, all_GT_flat_fine

def train(defModel, tokenizer, opti, schedulr, train_loader, dev_loader, val_loader, class_weights, args, device, exp_args):

    best_meanf1 = 0.0
    for epoch_i in range(0, args.max_eps):

        # put the model in train modeain()
        defModel.train()

        # Accumulate loss over an epoch
        total_train_loss = 0

        # (fine-grained) accumulate predictions and labels over the epoch
        train_epoch_logits_fine = []
        train_epochs_labels_fine = []

        for step, batch in enumerate(train_loader):
            # Clear the gradients
            opti.zero_grad()

            # Unpack this training batch from our dataloader. 
            b_input_ids_ = batch[0].cuda()
            with torch.cuda.device_of(b_input_ids_.data):
                b_input_ids = b_input_ids_.clone()

            # check if the model is GPT2-based
            if 'gpt2' in args.model_name:
                b_input_ids[b_input_ids_==0] = tokenizer.unk_token_id
                b_input_ids[b_input_ids_==101] = tokenizer.bos_token_id
                b_input_ids[b_input_ids_==102] = tokenizer.eos_token_id
           
            # fine grained entity labels
            b_f_input_mask = batch[4].cuda()
            b_f_labels = batch[5].cuda()

            # At this point you can either use coarse-grained or fine-grained labels
            b_loss_fine, b_output_fine, b_f_labels, b_mask = defModel(input_ids = b_input_ids, attention_mask=b_f_input_mask, labels=b_f_labels)
          
            total_train_loss += abs( b_loss_fine.item() )

            abs(b_loss_fine).backward()

            # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(defModel.parameters(), 1.0)

            #Optimization step
            opti.step()

            # Update the learning rate.
            schedulr.step()

            for i in range(0, b_f_labels.shape[0]): 

                selected_preds_fine = torch.masked_select(b_output_fine[i, ].cuda(), b_mask[i, ])
                selected_labs_fine = torch.masked_select(b_f_labels[i, ].cuda(), b_mask[i, ])

                train_epoch_logits_fine.extend( selected_preds_fine.to('cpu').numpy() )
                train_epochs_labels_fine.extend( selected_labs_fine.to("cpu").numpy() )

            if step % args.print_every == 0:
               
                cr_fine = classification_report(y_pred= train_epoch_logits_fine, y_true=train_epochs_labels_fine, labels= list(range(clf_P_fine_num_labels)), output_dict=True)
                if set_entity == 'participant':
                    meanF1_fine = ( cr_fine['1']['f1-score'] + cr_fine['2']['f1-score'] + cr_fine['3']['f1-score'] + cr_fine['4']['f1-score'] ) / 4
                elif set_entity == 'intervention':
                    meanF1_fine = ( cr_fine['1']['f1-score'] + cr_fine['2']['f1-score'] + cr_fine['3']['f1-score'] + cr_fine['4']['f1-score'] + cr_fine['5']['f1-score'] + cr_fine['6']['f1-score'] + cr_fine['7']['f1-score'] ) / 7
                elif set_entity == 'outcome':
                    meanF1_fine = ( cr_fine['1']['f1-score'] + cr_fine['2']['f1-score'] + cr_fine['3']['f1-score'] + cr_fine['4']['f1-score'] + cr_fine['5']['f1-score'] + cr_fine['6']['f1-score'] ) / 6

                print("Iteration {} of epoch {} complete. Loss : {:.4f} F1 mean : {:.8f}".format(step, epoch_i, abs ( b_loss_fine.item() ), meanF1_fine))


        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        print("  Average training loss: {0:.6f}".format(avg_train_loss))
        writer.add_scalar('loss-train-fine', avg_train_loss, epoch_i)

        train_cr_fine = classification_report(y_pred= train_epoch_logits_fine, y_true=train_epochs_labels_fine, labels= list(range(clf_P_fine_num_labels)), output_dict=True) 
        if set_entity == 'participant':
            train_meanF1_fine = ( train_cr_fine['1']['f1-score'] + train_cr_fine['2']['f1-score'] + train_cr_fine['3']['f1-score'] + train_cr_fine['4']['f1-score'] ) / 4
            writer.add_scalar('f1-train-fine', train_meanF1_fine, epoch_i) # write validation F1 score to logs
        elif set_entity == 'intervention':
            train_meanF1_fine = ( train_cr_fine['1']['f1-score'] + train_cr_fine['2']['f1-score'] + train_cr_fine['3']['f1-score'] + train_cr_fine['4']['f1-score'] + train_cr_fine['5']['f1-score'] + train_cr_fine['6']['f1-score'] + train_cr_fine['7']['f1-score'] ) / 7
            writer.add_scalar('f1-train-fine', train_meanF1_fine, epoch_i) # write validation F1 score to logs
        elif set_entity == 'outcome':
            train_meanF1_fine = ( train_cr_fine['1']['f1-score'] + train_cr_fine['2']['f1-score'] + train_cr_fine['3']['f1-score'] + train_cr_fine['4']['f1-score'] + train_cr_fine['5']['f1-score'] + train_cr_fine['6']['f1-score'] ) / 6
            writer.add_scalar('f1-train-fine', train_meanF1_fine, epoch_i) # write validation F1 score to logs      

        print('--------------------------------------------------------------------------------')
        print('Performing validation over the validation set...')
        print('--------------------------------------------------------------------------------')
        val_loss, val_cr_fine, all_pred_flat_fine, all_GT_flat_fine = evaluate_i(defModel, tokenizer, opti, schedulr, dev_loader, args, device, epoch_number = epoch_i)
        
        if set_entity == 'participant':
            val_meanF1_fine = ( val_cr_fine['1']['f1-score'] + val_cr_fine['2']['f1-score'] + val_cr_fine['3']['f1-score'] + val_cr_fine['4']['f1-score'] ) / 4
            writer.add_scalar('f1-validation-fine', val_meanF1_fine, epoch_i) # write validation F1 score to logs
        elif set_entity == 'intervention':
            val_meanF1_fine = ( val_cr_fine['1']['f1-score'] + val_cr_fine['2']['f1-score'] + val_cr_fine['3']['f1-score'] + val_cr_fine['4']['f1-score'] + val_cr_fine['5']['f1-score'] + val_cr_fine['6']['f1-score'] + val_cr_fine['7']['f1-score'] ) / 7
            writer.add_scalar('f1-validation-fine', val_meanF1_fine, epoch_i) # write validation F1 score to logs
        elif set_entity == 'outcome':
            val_meanF1_fine = ( val_cr_fine['1']['f1-score'] + val_cr_fine['2']['f1-score'] + val_cr_fine['3']['f1-score'] + val_cr_fine['4']['f1-score'] + val_cr_fine['5']['f1-score'] + val_cr_fine['6']['f1-score'] ) / 6
            writer.add_scalar('f1-validation-fine', val_meanF1_fine, epoch_i) # write validation F1 score to logs
        
        print("Epoch {} complete! Validation Loss : {}, Validation F1 : {:.8f}".format(epoch_i, abs(val_loss), val_meanF1_fine))

        if val_meanF1_fine > best_meanf1:

            print("Best validation mean F1 improved from {} to {} ...".format(best_meanf1, val_meanF1_fine ))
            
            # save the best model here
            model_name_here = '_stl_fine_' + set_entity +'/' + experiment_type + '/' + str(args.fold) + '/' + args.model_name + '_epoch_dropout' + str(epoch_i) + '_best_model.pt'
            print('Saving the best model for epoch {} with mean F1 score of {} '.format(epoch_i, val_meanF1_fine ))
            torch.save(defModel, model_name_here)
            saved_models.append(model_name_here)
            best_meanf1 = val_meanF1_fine   


if __name__ == "__main__":

    ##################################################################################
    # Load the BERT tokenizer XXX Set tokenizer
    ##################################################################################
    print('Loading BERT tokenizer...')
    if 'bert' in pretrained_model:
        tokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    elif 'gpt2' in pretrained_model:
        tokenizer_ = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True, unk_token="<|endoftext|>")


    ##################################################################################
    # Get the loaders
    ##################################################################################
    st = time.time()
    batch_size = 6
    max_length = 512
    
    train_keys, tokenized_train_, tokenized_train_fine_, tokenized_test, tokenized_test_fine, tokenized_test2, tokenized_test2_fine, class_weights_coarse, class_weights_fine = dl_individual(batch_size, max_length, tokenizer_, pretrained_model, experiment_type, exp_args) # individual

    class_weights_coarse = torch.FloatTensor(class_weights_coarse).to(device)
    class_weights_fine = torch.FloatTensor(class_weights_fine).to(device)
    print('Fine weights: ', class_weights_fine)
    print('Loaded the data in '.format(time.time() - st))

    ##########################################################################
    # Instead of random split use K-Fold function from Sklearn
    ##########################################################################
    kf = KFold(n_splits=10)

    for fold, (train_index, dev_index) in enumerate( kf.split(train_keys) ):

        if fold == 0:
                
            print('Working on the fold: ', fold)
            print( len(train_index), ' : ' , len(dev_index) )

            train_text_tokenized_, train_labels_tokenized_ = list(zip(*tokenized_train_))
            train_fine_text_tokenized_, train_fine_labels_tokenized_ = list(zip(*tokenized_train_fine_))

            train_text_tokenized_, train_labels_tokenized_ = list(zip(*tokenized_train_))
            train_fine_text_tokenized_, train_fine_labels_tokenized_ = list(zip(*tokenized_train_fine_))

            # Training data
            train_text_tokenized = np.array ( train_text_tokenized_ ) [train_index]
            train_fine_text_tokenized = np.array ( train_fine_text_tokenized_ ) [train_index]

            # Training labels
            train_labels_tokenized = np.array ( train_labels_tokenized_ ) [train_index]
            train_fine_labels_tokenized = np.array ( train_fine_labels_tokenized_ ) [train_index]

            # Development data
            dev_text_tokenized = np.array ( train_text_tokenized_ ) [dev_index]
            dev_fine_text_tokenized = np.array ( train_fine_text_tokenized_ ) [dev_index]

            # Development labels
            dev_labels_tokenized = np.array ( train_labels_tokenized_ ) [dev_index]
            dev_fine_labels_tokenized = np.array ( train_fine_labels_tokenized_ ) [dev_index]


            test_text_tokenized, test_labels_tokenized = list(zip(*tokenized_test))
            test_fine_text_tokenized, test_fine_labels_tokenized = list(zip(*tokenized_test_fine))

            test2_text_tokenized, test2_labels_tokenized = list(zip(*tokenized_test2))
            test2_fine_text_tokenized, test2_fine_labels_tokenized = list(zip(*tokenized_test2_fine))

            print('Creating attention masks ...')
            train_input_atten_mask = createAttnMask(train_text_tokenized)
            train_fine_input_atten_mask = createAttnMask(train_fine_text_tokenized)

            dev_input_atten_mask = createAttnMask(dev_text_tokenized)
            dev_fine_input_atten_mask = createAttnMask(dev_fine_text_tokenized)

            test_input_atten_mask = createAttnMask(test_text_tokenized)
            test_fine_input_atten_mask = createAttnMask(test_fine_text_tokenized)

            test2_input_atten_mask = createAttnMask(test2_text_tokenized)
            test2_fine_input_atten_mask = createAttnMask(test2_fine_text_tokenized)
            print('Attention masks created...')

            # Convert all inputs, labels, and attentions into torch tensors, the required datatype: torch.int64
            train_input_ids = torch.tensor(train_text_tokenized, dtype=torch.int64)
            train_fine_input_ids = torch.tensor(train_fine_text_tokenized, dtype=torch.int64)
            dev_input_ids = torch.tensor(dev_text_tokenized, dtype=torch.int64)
            dev_fine_input_ids = torch.tensor(dev_fine_text_tokenized, dtype=torch.int64)
            test_input_ids = torch.tensor(test_text_tokenized, dtype=torch.int64)
            test_fine_input_ids = torch.tensor(test_fine_text_tokenized, dtype=torch.int64)
            test2_input_ids = torch.tensor(test2_text_tokenized, dtype=torch.int64)
            test2_fine_input_ids = torch.tensor(test2_fine_text_tokenized, dtype=torch.int64)

            train_input_labels = torch.tensor(train_labels_tokenized, dtype=torch.int64)
            train_fine_input_labels = torch.tensor(train_fine_labels_tokenized, dtype=torch.int64)
            dev_input_labels = torch.tensor(dev_labels_tokenized, dtype=torch.int64)
            dev_fine_input_labels = torch.tensor(dev_fine_labels_tokenized, dtype=torch.int64)
            test_input_labels = torch.tensor(test_labels_tokenized, dtype=torch.int64)
            test_fine_input_labels = torch.tensor(test_fine_labels_tokenized, dtype=torch.int64)
            test2_input_labels = torch.tensor(test2_labels_tokenized, dtype=torch.int64)
            test2_fine_input_labels = torch.tensor(test2_fine_labels_tokenized, dtype=torch.int64)

            train_attention_mask = torch.tensor(train_input_atten_mask, dtype=torch.int64)
            train_fine_attention_mask = torch.tensor(train_fine_input_atten_mask, dtype=torch.int64)
            dev_attention_mask = torch.tensor(dev_input_atten_mask, dtype=torch.int64)
            dev_fine_attention_mask = torch.tensor(dev_fine_input_atten_mask, dtype=torch.int64)
            test_attention_mask = torch.tensor(test_input_atten_mask, dtype=torch.int64)
            test_fine_attention_mask = torch.tensor(test_fine_input_atten_mask, dtype=torch.int64)
            test2_attention_mask = torch.tensor(test2_input_atten_mask, dtype=torch.int64)
            test2_fine_attention_mask = torch.tensor(test2_fine_input_atten_mask, dtype=torch.int64)
            print('Inputs converted to tensors...')

            # Create the DataLoader for our training set.
            train_data = TensorDataset(train_input_ids, train_attention_mask, train_input_labels, train_fine_input_ids, train_fine_attention_mask, train_fine_input_labels)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=None, batch_size=batch_size, shuffle=False)

            # Create the DataLoader for our development set.
            dev_data = TensorDataset(dev_input_ids, dev_attention_mask, dev_input_labels, dev_fine_input_ids, dev_fine_attention_mask, dev_fine_input_labels)
            dev_sampler = RandomSampler(dev_data)
            development_dataloader = DataLoader(dev_data, sampler=None, batch_size=batch_size, shuffle=False)

            # Create the DataLoader for our hold-out test set (EBM-NLP).
            valid_data = TensorDataset(test_input_ids, test_attention_mask, test_input_labels, test_fine_input_ids, test_fine_attention_mask, test_fine_input_labels)
            valid_sampler = SequentialSampler(valid_data)
            validation_dataloader = DataLoader(valid_data, sampler=None, batch_size=batch_size, shuffle=False)

            # Create the DataLoader for our hold-out test set (from in-house corpus).
            valid2_data = TensorDataset(test2_input_ids, test2_attention_mask, test2_input_labels, test2_fine_input_ids, test2_fine_attention_mask, test2_fine_input_labels)
            valid2_sampler = SequentialSampler(valid2_data)
            validation2_dataloader = DataLoader(valid2_data, sampler=None, batch_size=batch_size, shuffle=False)

            print('\n--------------------------------------------------------------------------------------------')
            print('Data loaders created')
            print('--------------------------------------------------------------------------------------------') 

            # model to execute could be a single name or a list
            # model_names = ['bert_linear', 'bert_lstm_crf', 'bert_bilstm_crf', 'bert_lstm_attn_crf', 'bert_bilstm_attn_crf', 'gpt2_bilstm_crf', 'gpt2_bilstm_attn_crf']
            model_names = ['bert_bilstm_attn_crf']

            for model_name in model_names:
                
                print('Executing the model: ', model_name)

                # ##################################################################################
                # # Reinitialize and Load the tokenizer XXX Set tokenizer
                # ##################################################################################
                print('Loading BERT tokenizer...')
                if 'bert' in model_name:
                    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

                elif 'gpt2' in model_name:
                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True, unk_token="<|endoftext|>")


                ##################################################################################
                # Parse the arguments for training
                ##################################################################################
                parser = argparse.ArgumentParser()
                parser.add_argument('-gpu', type = int, default = device)
                parser.add_argument('-freeze_bert', action='store_false') # store_false = won't freeze BERT
                parser.add_argument('-print_every', type = int, default= 100)
                parser.add_argument('-max_eps', type = int, default= 15)
                parser.add_argument('-lr', type = float, default= 5e-5)
                parser.add_argument('-eps', type = float, default= 1e-8)
                parser.add_argument('-model_name', type = str, default = model_name)
                parser.add_argument('-loss', type = str, default = 'general')
                parser.add_argument('-fold', type = str, default=fold)
                args = parser.parse_args()

                PATH_SUM_WRITER = '_' + set_entity +'/' + experiment_type + '/' + str(args.fold) + '/' + args.model_name + '/'
                writer = SummaryWriter(PATH_SUM_WRITER) # XXX

                ##################################################################################
                #Instantiating the BERT model
                ##################################################################################
                print("Building model...")
                st = time.time()
            
                if model_name == 'bert_linear':
                    model = BERTLogReg(args.freeze_bert, tokenizer, class_weights_fine)

                elif model_name == 'bert_linear_crf':
                    model = BERTLogRegCRF(args.freeze_bert, tokenizer)

                elif model_name in ['bert_lstm_linear', 'bert_bilstm_linear']:
                    if 'bi' in model_name:
                        print('The model is bidirectional...')
                        bidirec = True
                    else:
                        bidirec = False
                    model = BERTLSTMLogReg(args.freeze_bert, tokenizer, device, bidirec, class_weights=class_weights_fine)

                elif model_name in ['bert_lstm_crf', 'bert_bilstm_crf']:
                    if 'bi' in model_name:
                        print('The model is bidirectional...')
                        bidirec = True
                    else:
                        bidirec = False
                    model = BERTLSTMLogRegCRF(args.freeze_bert, tokenizer, device, bidirec)

                elif model_name in ['bert_lstm_attn_crf', 'bert_bilstm_attn_crf']:
                    if 'bi' in model_name:
                        print('The model is bidirectional...')
                        bidirec = True
                    else:
                        bidirec = False
                    model = BERTLSTMattenLogRegCRF(args.freeze_bert, tokenizer, device, bidirec)

                elif model_name in ['bert_lstm_mattn_crf', 'bert_bilstm_mattn_crf']:
                    if 'bi' in model_name:
                        print('The model is bidirectional...')
                        bidirec = True
                    else:
                        bidirec = False
                    model = BERTLSTMMulattenLogRegCRF(args.freeze_bert, tokenizer, device, bidirec)

                elif model_name in ['bert_bilstm_mattn_clogreg']:
                    if 'bi' in model_name:
                        print('The model is bidirectional...')
                        bidirec = True
                    else:
                        bidirec = False
                    model = BERTLSTMMulattenfLogReg(args.freeze_bert, tokenizer, device, bidirec, class_weights = class_weights_fine)

                elif model_name in ['gpt2_lstm_crf', 'gpt2_bilstm_crf']:
                    if 'bi' in model_name:
                        print('The model is bidirectional...')
                        bidirec = True
                    else:
                        bidirec = False
                    model = GPT2LSTMLogRegCRF(args.freeze_bert, tokenizer, device, bidirec, class_weights_fine)

                elif model_name in ['gpt2_lstm_attn_crf', 'gpt2_bilstm_attn_crf']:
                    if 'bi' in model_name:
                        print('The model is bidirectional...')
                        bidirec = True
                    else:
                        bidirec = False
                    model = GPT2LSTMattenLogRegCRF(args.freeze_bert, tokenizer, device, bidirec)


                ##################################################################################
                # Tell pytorch to run data on this model on the GPU and parallelize it
                ##################################################################################
                # if torch.cuda.device_count() > 1:
                #     model = nn.DataParallel(model, device_ids = [0])
                #     print("Using", len(model.device_ids), " GPUs!")
                model.cuda()
                print('Total model parameters: ', count_parameters(model))
                print("Done in {} seconds".format(time.time() - st))

                ##################################################################################
                # Set up the optimizer and the scheduler
                ##################################################################################
                st = time.time()
                # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
                optimizer = AdamW(model.parameters(),
                                lr = args.lr, # args.learning_rate - default is 5e-5 (for BERT-base)
                                eps = args.eps, # args.adam_epsilon  - default is 1e-8.
                                )

                # Total number of training steps is number of batches * number of epochs.
                total_steps = len(train_dataloader) * args.max_eps
                print('Total steps per epoch: ', total_steps)

                # Create the learning rate scheduler.
                scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                            num_warmup_steps=0,
                                                            num_training_steps = total_steps)
                

                print("Created the optimizer, scheduler and loss function objects in {} seconds".format(time.time() - st))


                print('##################################################################################')
                print('Begin training...')
                print('##################################################################################')
                train(model, tokenizer, optimizer, scheduler, train_dataloader, development_dataloader, validation_dataloader, class_weights_fine, args, device, exp_args)
                print("Training and validation done in {} seconds".format(time.time() - st))

                print('##################################################################################')
                print('Begin test...')
                print('##################################################################################')
                test_model = torch.load( saved_models[-1] ) # or add the path to the model you'd like to test
                test_model.cuda()

                print('Applying the best model on test set (EBM-NLP)...')
                EBM_file = '_'+ set_entity + '/seq_lab/f1_scores_fine/ebm.txt'
                with open(EBM_file, 'a+') as f:
                    f.write(model_name)
                    f.write('\n')
                test_loss, test_cr_fine, all_pred_flat_fine, all_GT_flat_fine = evaluate_i(test_model, tokenizer, optimizer, scheduler, validation_dataloader, args, device, filewrite=EBM_file)

                if set_entity == 'participant':
                    test_meanF1_fine = ( test_cr_fine['1']['f1-score'] + test_cr_fine['2']['f1-score'] + test_cr_fine['3']['f1-score'] + test_cr_fine['4']['f1-score'] ) / 4
                    print('f1 score (fine-grained) on test set (EBM-NLP) for the best model: ', test_meanF1_fine)
                elif set_entity == 'intervention':
                    test_meanF1_fine = ( test_cr_fine['1']['f1-score'] + test_cr_fine['2']['f1-score'] + test_cr_fine['3']['f1-score'] + test_cr_fine['4']['f1-score'] + test_cr_fine['5']['f1-score'] + test_cr_fine['6']['f1-score'] + test_cr_fine['7']['f1-score'] ) / 7
                    print('f1 score (fine-grained) on test set (EBM-NLP) for the best model: ', test_meanF1_fine)
                elif set_entity == 'outcome':
                    test_meanF1_fine = ( test_cr_fine['1']['f1-score'] + test_cr_fine['2']['f1-score'] + test_cr_fine['3']['f1-score'] + test_cr_fine['4']['f1-score'] + test_cr_fine['5']['f1-score'] + test_cr_fine['6']['f1-score'] ) / 6
                    print('f1 score (fine-grained) on test set (EBM-NLP) for the best model: ', test_meanF1_fine)


                inhouse_file = '_'+ set_entity + '/seq_lab/f1_scores_fine/inhouse.txt'
                with open(inhouse_file, 'a+') as f:
                    f.write(model_name)
                    f.write('\n')
                print('Applying the best model on test set (in-house corpus)...')
                test_loss, test_cr_fine, all_pred_flat_fine, all_GT_flat_fine = evaluate_i(test_model, tokenizer, optimizer, scheduler, validation2_dataloader, args, device, filewrite=inhouse_file)

                if set_entity == 'participant':
                    test_meanF1_fine = ( test_cr_fine['1']['f1-score'] + test_cr_fine['2']['f1-score'] + test_cr_fine['3']['f1-score'] + test_cr_fine['4']['f1-score'] ) / 4
                    print('f1 score (fine-grained) on test set (in-house corpus) for the best model: ', test_meanF1_fine)
                elif set_entity == 'intervention':
                    test_meanF1_fine = ( test_cr_fine['1']['f1-score'] + test_cr_fine['2']['f1-score'] + test_cr_fine['3']['f1-score'] + test_cr_fine['4']['f1-score'] + test_cr_fine['5']['f1-score'] + test_cr_fine['6']['f1-score'] + test_cr_fine['7']['f1-score'] ) / 7
                    print('f1 score (fine-grained) on test set (in-house corpus) for the best model: ', test_meanF1_fine)
                elif set_entity == 'outcome':
                    test_meanF1_fine = ( test_cr_fine['1']['f1-score'] + test_cr_fine['2']['f1-score'] + test_cr_fine['3']['f1-score'] + test_cr_fine['4']['f1-score'] + test_cr_fine['5']['f1-score'] + test_cr_fine['6']['f1-score'] ) / 6
                    print('f1 score (fine-grained) on test set (in-house corpus) for the best model: ', test_meanF1_fine)
