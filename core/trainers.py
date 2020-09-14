import torch.nn as nn
import os
import torch
import math
import time
from tqdm import tqdm
from typing import cast, List, Optional, Tuple, TypeVar
from core.utils.tensors import to_device
import random

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
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir, '{}_{}.pth'.format(epoch,
                                                    'optimizer_checkpoint')))

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
        print(lm_loss)
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

    # def parse_batch(
    #         self,
    #         batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    #     inputs = to_device(batch[0], device=self.device)
    #     inputs_lens = to_device(batch[1], device=self.device)
    #     inputs_att = to_device(batch[2], device=self.device)
    #     targets = to_device(batch[3], device=self.device)
    #     targets_lens = to_device(batch[4], device=self.device)
    #     targets_att = to_device(batch[5], device=self.device)
    #     return inputs, inputs_att, targets, targets_att

    # def get_predictions_and_targets(
    #         self: TrainerType,
    #         batch: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    #     inputs, inputs_mask, targets, targets_mask = self.parse_batch(batch)
    #     y_pred = self.model(inputs, inputs_mask, targets, targets_mask)
    #     return y_pred, inputs2

    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0

            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)

                outputs = self.model(input_ids=inputs,
                                     attention_mask=inputs_att,
                                     labels=inputs)
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
            self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir, '{}_{}.pth'.format(epoch,
                                                    'optimizer_checkpoint')))

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)

        outputs = self.model(input_ids=inputs,
                             attention_mask=inputs_att,
                             labels=inputs)
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
                                     lm_labels=repl_targets)
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
            self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir, '{}_{}.pth'.format(epoch,
                                                    'optimizer_checkpoint')))

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
                             lm_labels=repl_targets)
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
            self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir, '{}_{}.pth'.format(epoch,
                                                    'optimizer_checkpoint')))

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
            self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))
        torch.save(self.optimizer.state_dict(), os.path.join(
            self.checkpoint_dir, '{}_{}.pth'.format(epoch,
                                                    'optimizer_checkpoint')))

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