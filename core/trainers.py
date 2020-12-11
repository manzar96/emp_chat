import torch.nn as nn
import os
import torch
import math
import time
from tqdm import tqdm
import torch.nn.functional as F
from typing import cast, List, Optional, Tuple, TypeVar
from core.utils.tensors import to_device
import random
from core.modules.loss import SequenceCrossEntropyLoss
TrainerType = TypeVar('TrainerType', bound='Trainer')


class EncoderDecoderTransformerTrainer:

    def __init__(self, model,
                 optimizer,
                 patience,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience


    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                padded_targets = to_device(batch[2], device=self.device)
                replaced_targets = to_device(batch[3], device=self.device)
                targets_att = to_device(batch[4], device=self.device)

                outputs = self.model(input_ids=inputs,
                                     attention_mask=inputs_att,
                                     decoder_input_ids=padded_targets,
                                     decoder_attention_mask=targets_att,
                                     labels=replaced_targets)
                lm_loss = outputs[0]
                pred_scores = outputs[1]
                last_hidden = outputs[2]
                avg_val_loss += lm_loss.item()

            avg_val_loss = avg_val_loss / len(val_loader)
            return avg_val_loss

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # torch.save(self.model.state_dict(), os.path.join(
        #     self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))

        # we use the proposed method for saving EncoderDecoder model
        self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir,'optimizer_checkpoint'))


    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        padded_targets = to_device(batch[2], device=self.device)
        replaced_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)

        # episis den eimai sigouros gia to ti prepei na dwsw san
        # decoder_input_ids (ta input ids i ta padded_targets??)

        outputs = self.model(input_ids=inputs,
                             attention_mask=inputs_att,
                             decoder_input_ids=padded_targets,
                             decoder_attention_mask=targets_att,
                             labels=replaced_targets)

        lm_loss = outputs[0]
        # print(lm_loss)
        pred_scores = outputs[1]
        last_hidden = outputs[2]
        return lm_loss, last_hidden

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience  = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                loss, _ = self.train_step(sample_batch)
                avg_train_loss += loss.item()
                loss.backward(retain_graph=False)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class GPT2TransformerTrainer:

    def __init__(self, model,
                 optimizer,
                 patience,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience

    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                replaced_targets = to_device(batch[2], device=self.device)

                outputs = self.model(input_ids=inputs,
                                     attention_mask=inputs_att,
                                     labels=replaced_targets)

                lm_loss = outputs[0]
                pred_scores = outputs[1]
                last_hidden = outputs[2]
                avg_val_loss += lm_loss.item()

            avg_val_loss = avg_val_loss / len(val_loader)
            return avg_val_loss

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir,'model_checkpoint'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir, 'optimizer_checkpoint'))

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        replaced_targets = to_device(batch[2], device=self.device)

        outputs = self.model(input_ids=inputs,
                             attention_mask=inputs_att,
                             labels=replaced_targets)
        lm_loss = outputs[0]
        pred_scores = outputs[1]
        last_hidden = outputs[2]
        return lm_loss, last_hidden

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience  = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                loss, _ = self.train_step(sample_batch)
                avg_train_loss += loss.item()
                loss.backward(retain_graph=False)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class T5TransformerTrainer:

    def __init__(self, model,
                 optimizer,
                 patience,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience

    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                pad_targets = to_device(batch[2], device=self.device)
                repl_targets = to_device(batch[3], device=self.device)
                targets_att = to_device(batch[4], device=self.device)

                outputs = self.model(input_ids=inputs,
                                     attention_mask=inputs_att,
                                     labels=repl_targets)
                lm_loss = outputs[0]
                pred_scores = outputs[1]
                last_hidden = outputs[2]
                avg_val_loss += lm_loss.item()

            avg_val_loss = avg_val_loss / len(val_loader)
            return avg_val_loss

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # torch.save(self.model.state_dict(), os.path.join(
        #     self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))
        # we use the proposed method for saving T5 model
        self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir,'optimizer_checkpoint'))

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        pad_targets = to_device(batch[2], device=self.device)
        repl_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)

        outputs = self.model(input_ids=inputs,
                             attention_mask=inputs_att,
                             labels=repl_targets)
        lm_loss = outputs[0]
        # print(lm_loss)
        pred_scores = outputs[1]
        last_hidden = outputs[2]
        return lm_loss, last_hidden

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience  = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                loss, _ = self.train_step(sample_batch)
                avg_train_loss += loss.item()
                loss.backward(retain_graph=False)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class T5TransformerTrainerMultitask:

    def __init__(self, model,
                 optimizer,
                 patience,
                 criterion,
                 auxilary_loss_weight,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience
        self.criterion = criterion
        self.aux_weight = auxilary_loss_weight

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, 'model_checkpoint'),
                   _use_new_zipfile_serialization=False)
        # we use the proposed method for saving T5 model
        # self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir,'optimizer_checkpoint'))



    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            avg_val_lm_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                pad_targets = to_device(batch[2], device=self.device)
                repl_targets = to_device(batch[3], device=self.device)
                targets_att = to_device(batch[4], device=self.device)
                emo_label = to_device(batch[5], device=self.device)

                outputs = self.model(emolabel=emo_label, input_ids=inputs,
                                     attention_mask=inputs_att,
                                     labels=repl_targets)
                lm_loss = outputs[0]
                lm_logits = outputs[1]
                clf_logits = outputs[2]
                clf_loss = self.criterion(clf_logits, emo_label)

                avg_val_loss = avg_val_loss + lm_loss + clf_loss
                avg_val_lm_loss = avg_val_lm_loss + lm_loss


            avg_val_loss = avg_val_loss / len(val_loader)
            avg_val_lm_loss = avg_val_lm_loss / len(val_loader)
            print("avg val loss {} ,   avg val lm_loss {}".format(
                avg_val_loss, avg_val_lm_loss))
            return avg_val_lm_loss


    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        pad_targets = to_device(batch[2], device=self.device)
        repl_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)
        emo_label = to_device(batch[5], device=self.device)

        outputs = self.model(emolabel=emo_label, input_ids=inputs,
                             attention_mask=inputs_att,
                             labels=repl_targets)
        lm_loss = outputs[0]
        lm_logits = outputs[1]
        clf_logits = outputs[2]
        clf_loss = self.criterion(clf_logits, emo_label)
        return lm_loss, clf_loss

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            iters=0
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            avg_train_lm_loss = 0

            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                lm_loss, clf_loss = self.train_step(sample_batch)

                loss = lm_loss + self.aux_weight*clf_loss
                avg_train_loss += loss.item()
                avg_train_lm_loss += lm_loss.item()
                loss.backward(retain_graph=False)
                iters += 1
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
                if iters%400==0:
                    print("lm_loss {},   clf_loss  {}".format(lm_loss.item(),
                                                              self.aux_weight*clf_loss.item()))
                    print("total loss {}".format(loss.item()))
                    print("Train lm loss {}".format(avg_train_lm_loss/iters))
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_train_lm_loss = avg_train_lm_loss / len(train_loader)
            print("avg train loss {} ".format(avg_train_loss))
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_lm_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class T5TransformerTrainerMultitaskTriple:

    def __init__(self, model,
                 optimizer,
                 patience,
                 criterion,
                 auxilary_loss_weight1,
                 auxilary_loss_weight2,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience
        self.criterion = criterion
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.pdist = nn.PairwiseDistance(p=2)
        # self.mse_loss = nn.MSELoss()
        self.aux_weight1 = auxilary_loss_weight1
        self.aux_weight2 = auxilary_loss_weight2

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, 'model_checkpoint'),
                   _use_new_zipfile_serialization=False)
        # we use the proposed method for saving T5 model
        # self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir,'optimizer_checkpoint'))



    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            avg_val_lm_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                pad_targets = to_device(batch[2], device=self.device)
                repl_targets = to_device(batch[3], device=self.device)
                targets_att = to_device(batch[4], device=self.device)
                emo_label = to_device(batch[5], device=self.device)

                outputs = self.model(emolabel=emo_label, input_ids=inputs,
                                     attention_mask=inputs_att,
                                     labels=repl_targets)
                lm_loss = outputs[0]
                lm_logits = outputs[1]
                clf_logits = outputs[2]
                clf_loss = self.criterion(clf_logits, emo_label)

                avg_val_loss = avg_val_loss + lm_loss + clf_loss
                avg_val_lm_loss = avg_val_lm_loss + lm_loss


            avg_val_loss = avg_val_loss / len(val_loader)
            avg_val_lm_loss = avg_val_lm_loss / len(val_loader)
            print("avg val loss {} ,   avg val lm_loss {}".format(
                avg_val_loss, avg_val_lm_loss))
            return avg_val_lm_loss


    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        pad_targets = to_device(batch[2], device=self.device)
        repl_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)
        emo_label = to_device(batch[5], device=self.device)

        outputs = self.model(emolabel=emo_label, input_ids=inputs,
                             attention_mask=inputs_att,
                             labels=repl_targets)
        lm_loss = outputs[0]
        lm_logits = outputs[1]
        clf_logits_enc = outputs[2]
        clf_logits_dec = outputs[3]

        clf_loss_enc = self.criterion(clf_logits_enc, emo_label)
        clf_loss_dec = self.criterion(clf_logits_dec, emo_label)

        return lm_loss, clf_loss_enc, clf_loss_dec

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            iters=0
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            avg_train_lm_loss = 0

            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                lm_loss, clf_loss_enc, clf_loss_dec = self.train_step(
                    sample_batch)

                loss = lm_loss + self.aux_weight1*clf_loss_enc + \
                       self.aux_weight2*clf_loss_dec
                avg_train_loss += loss.item()
                avg_train_lm_loss += lm_loss.item()
                loss.backward(retain_graph=False)
                iters += 1
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
                if iters%400==0:
                    print("lm_loss {},   clf_loss_enc  {} , clf_loss_dec {}".format(
                        lm_loss.item(), self.aux_weight1*clf_loss_enc.item(),
                        self.aux_weight2*clf_loss_dec.item()))
                    print("total loss {}".format(loss.item()))
                    print("Train lm loss {}".format(avg_train_lm_loss/iters))
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_train_lm_loss = avg_train_lm_loss / len(train_loader)
            print("avg train loss {}".format(avg_train_loss))
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_lm_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class BartTransformerTrainer:

    def __init__(self, model,
                 optimizer,
                 patience,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience
        self.crit = SequenceCrossEntropyLoss(pad_idx=-100)

    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                pad_targets = to_device(batch[2], device=self.device)
                repl_targets = to_device(batch[3], device=self.device)
                targets_att = to_device(batch[4], device=self.device)

                outputs = self.model(input_ids=inputs,
                                     attention_mask=inputs_att,
                                     decoder_input_ids=pad_targets)


                pred_scores = outputs[0]
                last_hidden = outputs[1]
                lm_loss = self.crit(pred_scores, repl_targets)
                avg_val_loss += lm_loss.item()

            avg_val_loss = avg_val_loss / len(val_loader)
            return avg_val_loss

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # torch.save(self.model.state_dict(), os.path.join(
        #     self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))
        # we use the proposed method for saving T5 model
        self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir,'optimizer_checkpoint'))

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        pad_targets = to_device(batch[2], device=self.device)
        repl_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)

        outputs = self.model(input_ids=inputs, attention_mask=inputs_att,
                             decoder_input_ids=pad_targets)

        pred_scores = outputs[0]
        last_hidden = outputs[1]
        lm_loss = self.crit(pred_scores, repl_targets)
        return lm_loss, last_hidden

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience  = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                loss, _ = self.train_step(sample_batch)
                avg_train_loss += loss.item()
                loss.backward(retain_graph=False)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)

class TransformerVaswaniTrainer:

    def __init__(self, model,
                 optimizer,
                 criterion,
                 patience,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience

    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                pad_targets = to_device(batch[2], device=self.device)
                repl_targets = to_device(batch[3], device=self.device)
                targets_att = to_device(batch[4], device=self.device)
                scores, preds, encoder_states = self.model(inputs,
                                                           ys=pad_targets)
                lm_loss = self.criterion(scores, pad_targets)
                avg_val_loss += lm_loss.item()

            avg_val_loss = avg_val_loss / len(val_loader)
            return avg_val_loss

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, 'model_checkpoint'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir, 'optimizer_checkpoint'))

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        pad_targets = to_device(batch[2], device=self.device)
        repl_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)
        scores, preds, encoder_states = self.model(inputs, ys=pad_targets)
        lm_loss = self.criterion(scores, pad_targets)
        return lm_loss, None

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience  = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                loss, _ = self.train_step(sample_batch)
                avg_train_loss += loss.item()
                loss.backward(retain_graph=False)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class TransformerVaswaniTrainerMultitask:

    def __init__(self, model,
                 optimizer,
                 criterion1,
                 criterion2,
                 patience,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience

    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss1 = 0
            avg_val_loss2 = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                pad_targets = to_device(batch[2], device=self.device)
                repl_targets = to_device(batch[3], device=self.device)
                targets_att = to_device(batch[4], device=self.device)
                emo_labels = to_device(batch[5], device=self.device)
                scores_lm, preds_lm, scores_clf, preds_clf, encoder_states = \
                    self.model(inputs, ys=pad_targets)
                lm_loss = self.criterion1(scores_lm, pad_targets)
                clf_loss = self.criterion2(scores_clf, emo_labels)
                avg_val_loss1 += lm_loss.item()
                avg_val_loss2 += clf_loss.item()

            avg_val_loss1 = avg_val_loss1 / len(val_loader)
            avg_val_loss2 = avg_val_loss2 / len(val_loader)
            return avg_val_loss1, avg_val_loss2

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    avg_train_epoch_loss1,avg_val_epoch_loss1,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss1)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss1)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, 'model_checkpoint'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir, 'optimizer_checkpoint'))

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        pad_targets = to_device(batch[2], device=self.device)
        repl_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)
        emo_labels = to_device(batch[5], device=self.device)
        scores_lm, preds_lm, scores_clf, preds_clf, encoder_states = \
            self.model(inputs, ys=pad_targets)
        lm_loss = self.criterion1(scores_lm, pad_targets)
        clf_loss = self.criterion2(scores_clf, emo_labels)
        return lm_loss, clf_loss

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience  = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            avg_train_loss1 = 0
            avg_train_loss2 = 0
            avg_train_loss = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                loss1, loss2 = self.train_step(sample_batch)
                loss = 0.8*loss1 + 0.2*loss2
                avg_train_loss1 += loss1.item()
                avg_train_loss2 += loss2.item()
                avg_train_loss += loss.item()
                loss.backward(retain_graph=False)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
            avg_train_loss1 = avg_train_loss1 / len(train_loader)
            avg_train_loss2 = avg_train_loss2 / len(train_loader)
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_val_loss1, avg_val_loss2 = self.calc_val_loss(val_loader)
            avg_val_loss = 0.8*avg_val_loss1 + 0.2 * avg_train_loss2
            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss,
                             avg_train_loss1,avg_val_loss1,
                             cur_patience, strt)


    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class T5TransformerTrainerSimilarity:

    def __init__(self, model,
                 optimizer,
                 patience,
                 criterion,
                 auxilary_loss_weight,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience
        self.criterion = criterion
        self.aux_weight = auxilary_loss_weight
        # self.similarity = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.similarity = nn.CosineEmbeddingLoss()

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, 'model_checkpoint.pth'),
                   _use_new_zipfile_serialization=False)
        # we use the proposed method for saving T5 model
        # self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))
        # torch.save(self.optimizer.state_dict(), os.path.join(
        #     self.checkpoint_dir,'optimizer_checkpoint'))



    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            avg_val_lm_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                pad_targets = to_device(batch[2], device=self.device)
                repl_targets = to_device(batch[3], device=self.device)
                targets_att = to_device(batch[4], device=self.device)
                emo_label = to_device(batch[5], device=self.device)

                outputs = self.model(emolabel=emo_label, input_ids=inputs,
                                     attention_mask=inputs_att,
                                     labels=repl_targets)
                lm_loss = outputs[0]
                lm_logits = outputs[1]
                clf_logits = outputs[2]
                enc_emo_repr = outputs[3]
                dec_emo_repr = outputs[5]
                # similarity_loss = torch.mean(self.similarity(enc_emo_repr,
                #                                         dec_emo_repr))
                similarity_loss = self.similarity(enc_emo_repr,
                                                  dec_emo_repr,
                                                  torch.tensor(1, device=self.device))
                clf_loss = self.criterion(clf_logits, emo_label)

                # avg_val_loss = avg_val_loss + lm_loss + clf_loss-similarity_loss
                avg_val_loss = avg_val_loss + lm_loss + clf_loss+similarity_loss
                avg_val_lm_loss = avg_val_lm_loss + lm_loss


            avg_val_loss = avg_val_loss / len(val_loader)
            avg_val_lm_loss = avg_val_lm_loss / len(val_loader)
            print("avg val loss {} ,   avg val lm_loss {}".format(
                avg_val_loss, avg_val_lm_loss))
            return avg_val_lm_loss


    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        pad_targets = to_device(batch[2], device=self.device)
        repl_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)
        emo_label = to_device(batch[5], device=self.device)

        outputs = self.model(emolabel=emo_label, input_ids=inputs,
                             attention_mask=inputs_att,
                             labels=repl_targets)

        lm_loss = outputs[0]
        lm_logits = outputs[1]
        clf_logits_enc = outputs[2]
        enc_emo_repr = outputs[3]
        dec_emo_repr = outputs[5]
        # similarity_loss = torch.mean(self.similarity(enc_emo_repr,
        #                                              dec_emo_repr))
        similarity_loss = self.similarity(enc_emo_repr, dec_emo_repr,
                          torch.tensor(1, device=self.device))
        clf_loss = self.criterion(clf_logits_enc, emo_label)
        return lm_loss, clf_loss, similarity_loss

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            iters=0
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            avg_train_lm_loss = 0
            avg_clf_loss = 0
            avg_similarity = 0

            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                lm_loss, clf_loss, similarity_loss = self.train_step(
                    sample_batch)

                # loss = lm_loss + self.aux_weight*clf_loss - similarity_loss
                loss = lm_loss + 0.8*self.aux_weight*clf_loss + \
                                   0.6*similarity_loss
                avg_train_loss += loss.item()
                avg_clf_loss += clf_loss.item()
                avg_similarity += similarity_loss.item()
                avg_train_lm_loss += lm_loss.item()
                loss.backward(retain_graph=False)
                iters += 1
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
                if iters%400==0:
                    print("lm_loss {},   clf_loss  {}".format(lm_loss.item(),
                                                              0.8*clf_loss.item()))
                    # print("total loss {}".format(loss.item()))
                    # print("Train lm loss {}".format(avg_train_lm_loss/iters))
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_train_lm_loss = avg_train_lm_loss / len(train_loader)
            avg_clf_loss = avg_clf_loss / len(train_loader)
            avg_similarity = avg_similarity / len(train_loader)
            print("avg train loss {} ".format(avg_train_loss))
            print("avg train clf loss {} ".format(avg_clf_loss))
            print("avg train similarity loss {} ".format(avg_similarity))
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_lm_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class T5TransformerTrainerNeg:

    def __init__(self, model,
                 optimizer,
                 patience,
                 criterion,
                 multitask1=1.0,
                 multitask2=1.0,
                 margin=1.0,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience
        self.criterion = criterion
        self.similarity = nn.TripletMarginLoss(margin=margin,p=2)
        self.multitask1 = multitask1
        self.multitask2 = multitask2

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, 'model_checkpoint.pth'),
                   _use_new_zipfile_serialization=False)
        # we use the proposed method for saving T5 model
        # self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))
        # torch.save(self.optimizer.state_dict(), os.path.join(
        #     self.checkpoint_dir,'optimizer_checkpoint'))



    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            avg_val_lm_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                pad_targets = to_device(batch[2], device=self.device)
                repl_targets = to_device(batch[3], device=self.device)
                targets_att = to_device(batch[4], device=self.device)
                emo_label = to_device(batch[5], device=self.device)

                outputs = self.model.forward_validate(emolabel=emo_label,
                                            input_ids=inputs,
                                     attention_mask=inputs_att,
                                     labels=repl_targets)
                lm_loss = outputs[0]
                lm_logits = outputs[1]
                clf_logits = outputs[2]
                enc_emo_repr = outputs[3]
                dec_emo_repr = outputs[5]
                # similarity_loss = torch.mean(self.similarity(enc_emo_repr,
                #                                         dec_emo_repr))
                similarity_loss = self.similarity(enc_emo_repr,
                                                  dec_emo_repr,
                                                  torch.tensor(1, device=self.device))
                clf_loss = self.criterion(clf_logits, emo_label)

                # avg_val_loss = avg_val_loss + lm_loss + clf_loss-similarity_loss
                avg_val_loss = avg_val_loss + lm_loss + clf_loss+similarity_loss
                avg_val_lm_loss = avg_val_lm_loss + lm_loss


            avg_val_loss = avg_val_loss / len(val_loader)
            avg_val_lm_loss = avg_val_lm_loss / len(val_loader)
            print("avg val loss {} ,   avg val lm_loss {}".format(
                avg_val_loss, avg_val_lm_loss))
            return avg_val_lm_loss


    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        pad_targets = to_device(batch[2], device=self.device)
        repl_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)
        emo_label = to_device(batch[5], device=self.device)
        neg_padded = to_device(batch[6], device=self.device)
        neg_att = to_device(batch[7], device=self.device)

        outputs = self.model(emolabel=emo_label,
                             neg_ids=neg_padded,
                             input_ids=inputs,
                             attention_mask=inputs_att,
                             labels=repl_targets)

        lm_loss = outputs[0]
        lm_logits = outputs[1]
        clf_logits_enc = outputs[2]
        enc_emo_repr = outputs[3]
        clf_logits_dec = outputs[4]
        dec_emo_repr = outputs[5]
        dec_emo_repr_neg = outputs[7]
        similarity_loss = self.similarity(dec_emo_repr, enc_emo_repr,dec_emo_repr_neg)
        clf_loss = self.criterion(clf_logits_enc, emo_label)
        enc_emo = F.softmax(clf_logits_enc,dim=1)
        enc_emo = torch.argmax(enc_emo, dim=1)
        dec_emo = F.softmax(clf_logits_dec,dim=1)
        dec_emo = torch.argmax(dec_emo, dim=1)

        enc_acc = sum(enc_emo==emo_label).item() / inputs.shape[0]
        dec_acc = sum(enc_emo==dec_emo).item() / inputs.shape[0]
        return lm_loss, clf_loss, similarity_loss,enc_acc,dec_acc

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            iters=0
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            avg_train_lm_loss = 0
            avg_clf_loss = 0
            avg_similarity = 0
            avg_enc_acc = 0
            avg_dec_acc = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                lm_loss, clf_loss, similarity_loss,enc_acc,dec_acc = \
                    self.train_step(
                    sample_batch)

                # loss = lm_loss + self.aux_weight*clf_loss - similarity_loss
                loss = lm_loss + self.multitask1*clf_loss + \
                                   self.multitask2*similarity_loss
                avg_train_loss += loss.item()
                avg_clf_loss += self.multitask1*clf_loss.item()
                avg_similarity += self.multitask2*similarity_loss.item()
                avg_train_lm_loss += lm_loss.item()
                avg_enc_acc += enc_acc
                avg_dec_acc += dec_acc
                loss.backward(retain_graph=False)
                iters += 1
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
                if iters%400==0:
                    print("lm_loss {},   clf_loss  {}".format(lm_loss.item(),
                                                              self.multitask1*clf_loss.item()))
                    # print("total loss {}".format(loss.item()))
                    # print("Train lm loss {}".format(avg_train_lm_loss/iters))
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_train_lm_loss = avg_train_lm_loss / len(train_loader)
            avg_clf_loss = avg_clf_loss / len(train_loader)
            avg_similarity = avg_similarity / len(train_loader)
            avg_enc_acc = avg_enc_acc / len(train_loader)
            avg_dec_acc = avg_dec_acc / len(train_loader)

            print("avg train loss {} ".format(avg_train_loss))
            print("avg train clf loss {} ".format(avg_clf_loss))
            print("avg train similarity loss {} ".format(avg_similarity))
            print("avg enc acc {} ".format(avg_enc_acc))
            print("avg dec acc {} ".format(avg_dec_acc))
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_lm_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class BlenderBotTrainer:

    def __init__(self, model,
                 optimizer,
                 patience,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience
        self.crit = SequenceCrossEntropyLoss(pad_idx=-100)

    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                pad_targets = to_device(batch[2], device=self.device)
                repl_targets = to_device(batch[3], device=self.device)
                targets_att = to_device(batch[4], device=self.device)

                outputs = self.model(input_ids=inputs,
                                     attention_mask=inputs_att,
                                     labels=pad_targets, return_dict=True)

                lm_loss = outputs['loss']
                lm_logits = outputs['logits']
                last_hidden = outputs['decoder_hidden_states'][1]
                avg_val_loss += lm_loss.item()

            avg_val_loss = avg_val_loss / len(val_loader)
            return avg_val_loss

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} | Train PPL: {}".format(
            avg_train_epoch_loss, math.exp(avg_train_epoch_loss)))
        print("Val loss: {} | Val PPL: {}".format(avg_val_epoch_loss,
              math.exp(avg_val_epoch_loss)))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # torch.save(self.model.state_dict(), os.path.join(
        #     self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))
        # we use the proposed method for saving T5 model
        self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir,'optimizer_checkpoint'))

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        pad_targets = to_device(batch[2], device=self.device)
        repl_targets = to_device(batch[3], device=self.device)
        targets_att = to_device(batch[4], device=self.device)
        outputs = self.model(input_ids=inputs, attention_mask=inputs_att,
                             labels=repl_targets, return_dict=True)

        lm_loss = outputs['loss']
        lm_logits = outputs['logits']
        last_hidden = outputs['decoder_hidden_states'][1]

        return lm_loss, last_hidden

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_loss, cur_patience  = 10000, 0

        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                loss, _ = self.train_step(sample_batch)
                avg_train_loss += loss.item()
                loss.backward(retain_graph=False)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_val_loss = self.calc_val_loss(val_loader)

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)