import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import shutil

from DRL.actor import *
from Renderer.stroke_gen import *
from Renderer.model import *


class Actor(object):
    def __init__(self, img_name=None, resultView=None):
        assert resultView is not None
        self.resultView = resultView

        if os.path.isdir('output'):
            shutil.rmtree('output')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width = 128

        parser = argparse.ArgumentParser(description='Learning to Paint')
        parser.add_argument('--max_step', default=100, type=int, help='max length for episode')
        parser.add_argument('--actor', default='../pkl/actor.pkl', type=str, help='Actor model')
        parser.add_argument('--renderer', default='../pkl/renderer.pkl', type=str, help='renderer model')
        parser.add_argument('--img', default='../image/mayun.png', type=str, help='test image')
        parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
        parser.add_argument('--divide', default=4, type=int, help='divide the target image to get better resolution')
        self.args = parser.parse_args()

        if img_name is not None:
            self.args.img = img_name

        self.canvas_cnt = self.args.divide * self.args.divide
        self.T = torch.ones([1, 1, self.width, self.width], dtype=torch.float32).to(self.device)
        self.img = cv2.imread(self.args.img, cv2.IMREAD_COLOR)
        self.origin_shape = (self.img.shape[1], self.img.shape[0])

        coord = torch.zeros([1, 2, self.width, self.width])
        for i in range(self.width):
            for j in range(self.width):
                coord[0, 0, i, j] = i / (self.width - 1.)
                coord[0, 1, i, j] = j / (self.width - 1.)
        self.coord = coord.to(self.device)  # Coordconv

        self.Decoder = FCN()
        self.Decoder.load_state_dict(torch.load(self.args.renderer))
        self.Decoder = self.Decoder.to(self.device).eval()

        self.actor = ResNet(9, 18, 65)  # action_bundle = 5, 65 = 5 * 13
        self.actor.load_state_dict(torch.load(self.args.actor))
        self.actor = self.actor.to(self.device).eval()

        self.canvas = torch.zeros([1, 3, self.width, self.width]).to(self.device)

        patch_img = cv2.resize(self.img, (self.width * self.args.divide, self.width * self.args.divide))
        patch_img = self.large2small(patch_img)
        patch_img = np.transpose(patch_img, (0, 3, 1, 2))
        self.patch_img = torch.tensor(patch_img).to(self.device).float() / 255.

        img = cv2.resize(self.img, (self.width, self.width))
        img = img.reshape(1, self.width, self.width, 3)
        img = np.transpose(img, (0, 3, 1, 2))
        self.img = torch.tensor(img).to(self.device).float() / 255.

        os.system('mkdir output')

    def decode(self, x, canvas):  # b * (10 + 3)
        x = x.view(-1, 10 + 3)
        stroke = 1 - self.Decoder(x[:, :10])
        stroke = stroke.view(-1, self.width, self.width, 1)
        color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
        stroke = stroke.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)
        stroke = stroke.view(-1, 5, 1, self.width, self.width)
        color_stroke = color_stroke.view(-1, 5, 3, self.width, self.width)
        res = []
        for i in range(5):
            canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
            res.append(canvas)
        return canvas, res

    def small2large(self, x):
        # (d * d, width, width) -> (d * width, d * width)
        x = x.reshape(self.args.divide, self.args.divide, self.width, self.width, -1)
        x = np.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape(self.args.divide * self.width, self.args.divide * self.width, -1)
        return x

    def large2small(self, x):
        # (d * width, d * width) -> (d * d, width, width)
        x = x.reshape(self.args.divide, self.width, self.args.divide, self.width, 3)
        x = np.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape(self.canvas_cnt, self.width, self.width, 3)
        return x

    def smooth(self, img):
        def smooth_pix(img, tx, ty):
            if tx == self.args.divide * self.width - 1 or ty == self.args.divide * self.width - 1 or tx == 0 or ty == 0:
                return img
            img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[
                tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
            return img

        for p in range(self.args.divide):
            for q in range(self.args.divide):
                x = p * self.width
                y = q * self.width
                for k in range(self.width):
                    img = smooth_pix(img, x + k, y + self.width - 1)
                    if q != self.args.divide - 1:
                        img = smooth_pix(img, x + k, y + self.width)
                for k in range(self.width):
                    img = smooth_pix(img, x + self.width - 1, y + k)
                    if p != self.args.divide - 1:
                        img = smooth_pix(img, x + self.width, y + k)
        return img

    def save_img(self, res, imgid, divide=False):
        output = res.detach().cpu().numpy()  # d * d, 3, width, width
        output = np.transpose(output, (0, 2, 3, 1))
        if divide:
            output = self.small2large(output)
            output = self.smooth(output)
        else:
            output = output[0]
        output = (output * 255).astype('uint8')
        output = cv2.resize(output, self.origin_shape)
        cv2.imwrite('output/generated' + str(imgid) + '.png', output)

    def act(self):
        with torch.no_grad():
            if self.args.divide != 1:
                self.args.max_step = self.args.max_step // 2
            for i in range(self.args.max_step):
                stepnum = self.T * i / self.args.max_step
                actions = self.actor(torch.cat([self.canvas, self.img, stepnum, self.coord], 1))
                self.canvas, res = self.decode(actions, self.canvas)
                # print('canvas step {}, L2Loss = {}'.format(i, ((self.canvas - self.img) ** 2).mean()))
                self.resultView.appendPlainText(
                    'canvas step {}, L2Loss = {}'.format(i, ((self.canvas - self.img) ** 2).mean()))
                for j in range(5):
                    self.save_img(res[j], self.args.imgid)
                    self.args.imgid += 1
            if self.args.divide != 1:
                self.canvas = self.canvas[0].detach().cpu().numpy()
                self.canvas = np.transpose(self.canvas, (1, 2, 0))
                self.canvas = cv2.resize(self.canvas, (self.width * self.args.divide, self.width * self.args.divide))
                self.canvas = self.large2small(self.canvas)
                self.canvas = np.transpose(self.canvas, (0, 3, 1, 2))
                self.canvas = torch.tensor(self.canvas).to(self.device).float()
                self.coord = self.coord.expand(self.canvas_cnt, 2, self.width, self.width)
                self.T = self.T.expand(self.canvas_cnt, 1, self.width, self.width)
                for i in range(self.args.max_step):
                    stepnum = self.T * i / self.args.max_step
                    actions = self.actor(torch.cat([self.canvas, self.patch_img, stepnum, self.coord], 1))
                    self.canvas, res = self.decode(actions, self.canvas)
                    # print('divided canvas step {}, L2Loss = {}'.format(i, ((self.canvas - self.patch_img) ** 2).mean()))
                    self.resultView.appendPlainText(
                        'divided canvas step {}, L2Loss = {}'.format(i, ((self.canvas - self.patch_img) ** 2).mean()))
                    for j in range(5):
                        self.save_img(res[j], self.args.imgid, True)
                        self.args.imgid += 1
