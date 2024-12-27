import argparse
import torch
import os

import numpy as np

from dataset import CHARS, LPRDataset
from model import LPRNet
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to val net')
    parser.add_argument('--model', default="weights/best.pth", help='the image size')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--val_img_dirs', default="datasets/combine/val", help='the val images path') # datasets/val
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--test_batch_size', default=100, help='testing batch size.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')

    args = parser.parse_args()

    return args

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


def decode(labelc):
    lb = ""
    for i in labelc:
        lb += CHARS[i]
    return lb

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
                print(f'{decode(target)} -> {decode(label)}')
                Tn_1 += 1
                continue
            if (np.asarray(target) == np.asarray(label)).all():
                Tp += 1
            else:
                print(f'{decode(target)} -> {decode(label)}')
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    return Acc

def val(args):
    lprnet = LPRNet(class_num=len(CHARS))
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")
    
    lprnet.load_state_dict(torch.load(args.model, weights_only=True))
    lprnet.eval()
    
    val_img_dirs = os.path.expanduser(args.val_img_dirs)
    val_dataset = LPRDataset(val_img_dirs.split(','), args.img_size, args.lpr_max_len)
    acc = Greedy_Decode_Eval(lprnet, val_dataset, args)
    print(acc)
    

if __name__ == "__main__":
    args = get_parser()
    val(args)