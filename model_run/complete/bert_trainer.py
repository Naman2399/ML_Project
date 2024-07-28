import argparse
import os

import torch
from torch.optim import Adam
import tqdm
from models.bert import ScheduledOptim
from utils.checkpoints import save_checkpoint, create_checkpoint_filename


class BERTTrainer:
    def __init__(
            self,
            model,
            train_dataloader,
            writer,
            args,
            ckpt_path,
            test_dataloader=None,
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            warmup_steps=10000,
            log_freq=10,
            device='cuda',
    ):

        self.device = device
        self.model = model
        self.model = model.to(device)
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.writer = writer
        self.args = args
        self.ckpt_path = ckpt_path

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps=warmup_steps
        )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.epoch = epoch
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.epoch = epoch
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        mode = "train" if train else "test"

        # Add model to cuda
        self.model = self.model.to(self.device)

        # progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )


        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            next_loss = self.criterion(next_sent_output, data["is_next"])

            # 2-2. NLLLoss of predicting masked token word
            # transpose to (m, vocab_size, seq_len) vs (m, seq_len)
            # criterion(mask_lm_output.view(-1, mask_lm_output.size(-1)), data["bert_label"].view(-1))
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = next_loss + mask_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        # Adding details in tensorboard
        self.writer.add_scalar('train/loss', avg_loss / len(data_loader))
        self.writer.add_scalar('train/acc', total_correct / total_element * 100)

        print(
            f"EP{epoch}, {mode}: \
            avg_loss={avg_loss / len(data_iter)}, \
            total_acc={total_correct * 100.0 / total_element}"
        )

        # Save the model at the end of every epoch
        save_checkpoint(self.args, self.model, self.optim, self.epoch, self.ckpt_path, avg_loss / len(data_loader))



def run(args : argparse.ArgumentParser, model, train_dataloader) :

    # Update ckpts file name
    checkpoint_filename = create_checkpoint_filename(args)
    checkpoint_path = os.path.join(args.ckpt_path, checkpoint_filename)

    bert_trainer = BERTTrainer(model, train_dataloader, writer=args.writer, args=args, ckpt_path= checkpoint_path,
                               device=args.device)
    epochs = 40

    for epoch in range(epochs) :
        bert_trainer.train(epoch)

    return