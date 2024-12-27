import torch
import argparse
import cv2

import numpy as np

from model import LPRNet
from dataset import CHARS

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to test')
    parser.add_argument('--img', required=True, type=str, help='test image path')
    
    return parser.parse_args()

def inference(args):
    model = LPRNet(class_num=len(CHARS))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load("weights/best.pth", weights_only=True))
    
    image = cv2.imread(args.img)
    image = cv2.resize(image, [94, 24])
    image = image.astype('float32')
    image -= 127.5
    image *= 0.0078125
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.from_numpy(image)
    inputs = image_tensor.unsqueeze(0)
    inputs = inputs.to(device)
    
    prebs = model(inputs)
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
    label = preb_labels[0]
    lb = ""
    for i in label:
        lb += CHARS[i]
    print(lb)


if __name__ == "__main__":
    args = get_parser()
    inference(args)