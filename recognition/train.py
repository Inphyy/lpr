import os
import argparse
import torch

import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch.amp import GradScaler, autocast
from dataset import CHARS, LPRDataLoader
from model import LPRNet
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch import optim

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch', default=100, type=int, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default="../datasets/combine/train", help='the train images path')
    parser.add_argument('--test_img_dirs', default="../datasets/combine/val", help='the test images path')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=128, help='training batch size.')
    parser.add_argument('--test_batch_size', default=128, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    parser.add_argument('--pretrained_model', default=False, type=bool, help='pretrained base model')

    args = parser.parse_args()

    return args

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int64)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def train():
    args = get_parser()

    T_length = 18

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    lprnet = LPRNet(class_num=len(CHARS))
    if args.pretrained_model:
        print("load pretrain model!")
        lprnet.load_state_dict(torch.load("./weights/best_pre.pth"))
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    optimizer = optim.Adam(lprnet.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # lf = lambda e: (((1 + math.cos(e * math.pi / args.max_epoch)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = GradScaler(device.type)
    gradient_noise_scale = 0.001

    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)

    acc = 0
    for epoch in range(args.max_epoch):
        lprnet.train()
        dataloader_with_progress = tqdm(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn), desc=f"Train:{epoch} Progress")
        epoch_loss = 0
        for images, labels, lengths in dataloader_with_progress:
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)

            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with autocast(device.type):
                logits = lprnet(images)
                log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
                log_probs = log_probs.log_softmax(2).requires_grad_()
                loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            scaler.scale(loss).backward()
            for param in lprnet.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * gradient_noise_scale
                    param.grad.data.add_(noise)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"epoch:{epoch}, loss:{epoch_loss}")

        lprnet.eval()
        test_acc = Greedy_Decode_Eval(lprnet, test_dataset, args)
        if test_acc > acc:
            acc = test_acc
            torch.save(lprnet.state_dict(), f"{args.save_folder}best.pth")
        print(f"best acc: {acc}")

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')

def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    # epoch_size = len(datasets) // args.test_batch_size
    # batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
    dataloader = DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    # for i in range(epoch_size):
    for images, labels, lengths in dataloader:
        # load train data
        # images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label)
            start += length
        # targets = np.array([el.numpy() for el in targets])

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            target = targets[i].numpy()
            if len(label) != len(target):
                Tn_1 += 1
                continue
            if (np.asarray(target) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    return Acc


if __name__ == "__main__":
    train()
