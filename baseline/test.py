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

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QPlainTextEdit
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time


class Window(QWidget):

    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)

        btn1 = QPushButton('选 择 图 片', self)
        btn1.setObjectName('ClickBtn')
        btn1.clicked.connect(self.openimage)

        btn2 = QPushButton('开 始 绘 制', self)
        btn2.setObjectName('ClickBtn2')
        btn2.clicked.connect(self.generate)

        btn3 = QPushButton('生 成 视 频', self)
        btn3.setObjectName('ClickBtn3')
        btn3.clicked.connect(self.generate_video)

        btn4 = QPushButton('播 放 视 频', self)
        btn4.setObjectName('ClickBtn4')
        btn4.clicked.connect(self.play_video)

        layout.addWidget(btn1)
        layout.addWidget(btn2)
        layout.addWidget(btn3)
        layout.addWidget(btn4)

        self.resultView = QPlainTextEdit(self)
        self.resultView.setReadOnly(True)

        layout.addWidget(self.resultView)
        self.imgName = None

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "All Files(*)")
        self.resultView.appendPlainText(os.path.basename(imgName))
        self.imgName = imgName

    def generate(self):
        # os.system('python test_bak.py --img {}'.format(self.imgName))
        if os.path.isdir('output'):
            shutil.rmtree('output')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        width = 128

        parser = argparse.ArgumentParser(description='Learning to Paint')
        parser.add_argument('--max_step', default=100, type=int, help='max length for episode')
        parser.add_argument('--actor', default='../pkl/actor.pkl', type=str, help='Actor model')
        parser.add_argument('--renderer', default='../pkl/renderer.pkl', type=str, help='renderer model')
        # parser.add_argument('--img', default='../image/mayun.png', type=str, help='test image')
        parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
        parser.add_argument('--divide', default=4, type=int, help='divide the target image to get better resolution')
        args = parser.parse_args()
        args.img = self.imgName

        canvas_cnt = args.divide * args.divide
        T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
        img = cv2.imread(args.img, cv2.IMREAD_COLOR)
        origin_shape = (img.shape[1], img.shape[0])

        coord = torch.zeros([1, 2, width, width])
        for i in range(width):
            for j in range(width):
                coord[0, 0, i, j] = i / (width - 1.)
                coord[0, 1, i, j] = j / (width - 1.)
        coord = coord.to(device)  # Coordconv

        Decoder = FCN()
        Decoder.load_state_dict(torch.load(args.renderer))

        def decode(x, canvas):  # b * (10 + 3)
            x = x.view(-1, 10 + 3)
            stroke = 1 - Decoder(x[:, :10])
            stroke = stroke.view(-1, width, width, 1)
            color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
            stroke = stroke.permute(0, 3, 1, 2)
            color_stroke = color_stroke.permute(0, 3, 1, 2)
            stroke = stroke.view(-1, 5, 1, width, width)
            color_stroke = color_stroke.view(-1, 5, 3, width, width)
            res = []
            for i in range(5):
                canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
                res.append(canvas)
            return canvas, res

        def small2large(x):
            # (d * d, width, width) -> (d * width, d * width)
            x = x.reshape(args.divide, args.divide, width, width, -1)
            x = np.transpose(x, (0, 2, 1, 3, 4))
            x = x.reshape(args.divide * width, args.divide * width, -1)
            return x

        def large2small(x):
            # (d * width, d * width) -> (d * d, width, width)
            x = x.reshape(args.divide, width, args.divide, width, 3)
            x = np.transpose(x, (0, 2, 1, 3, 4))
            x = x.reshape(canvas_cnt, width, width, 3)
            return x

        def smooth(img):
            def smooth_pix(img, tx, ty):
                if tx == args.divide * width - 1 or ty == args.divide * width - 1 or tx == 0 or ty == 0:
                    return img
                img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] +
                               img[
                                   tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[
                                   tx + 1, ty + 1]) / 9
                return img

            for p in range(args.divide):
                for q in range(args.divide):
                    x = p * width
                    y = q * width
                    for k in range(width):
                        img = smooth_pix(img, x + k, y + width - 1)
                        if q != args.divide - 1:
                            img = smooth_pix(img, x + k, y + width)
                    for k in range(width):
                        img = smooth_pix(img, x + width - 1, y + k)
                        if p != args.divide - 1:
                            img = smooth_pix(img, x + width, y + k)
            return img

        def save_img(res, imgid, divide=False):
            output = res.detach().cpu().numpy()  # d * d, 3, width, width
            output = np.transpose(output, (0, 2, 3, 1))
            if divide:
                output = small2large(output)
                output = smooth(output)
            else:
                output = output[0]
            output = (output * 255).astype('uint8')
            output = cv2.resize(output, origin_shape)
            cv2.imwrite('output/generated' + str(imgid) + '.png', output)

        actor = ResNet(9, 18, 65)  # action_bundle = 5, 65 = 5 * 13
        actor.load_state_dict(torch.load(args.actor))
        actor = actor.to(device).eval()
        Decoder = Decoder.to(device).eval()

        canvas = torch.zeros([1, 3, width, width]).to(device)

        patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
        patch_img = large2small(patch_img)
        patch_img = np.transpose(patch_img, (0, 3, 1, 2))
        patch_img = torch.tensor(patch_img).to(device).float() / 255.

        img = cv2.resize(img, (width, width))
        img = img.reshape(1, width, width, 3)
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.tensor(img).to(device).float() / 255.

        os.system('mkdir output')

        with torch.no_grad():
            if args.divide != 1:
                args.max_step = args.max_step // 2
            for i in range(args.max_step):
                stepnum = T * i / args.max_step
                actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
                canvas, res = decode(actions, canvas)
                # print('canvas step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))
                self.resultView.appendPlainText('canvas step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))
                QApplication.processEvents()
                for j in range(5):
                    save_img(res[j], args.imgid)
                    args.imgid += 1
            if args.divide != 1:
                canvas = canvas[0].detach().cpu().numpy()
                canvas = np.transpose(canvas, (1, 2, 0))
                canvas = cv2.resize(canvas, (width * args.divide, width * args.divide))
                canvas = large2small(canvas)
                canvas = np.transpose(canvas, (0, 3, 1, 2))
                canvas = torch.tensor(canvas).to(device).float()
                coord = coord.expand(canvas_cnt, 2, width, width)
                T = T.expand(canvas_cnt, 1, width, width)
                for i in range(args.max_step):
                    stepnum = T * i / args.max_step
                    actions = actor(torch.cat([canvas, patch_img, stepnum, coord], 1))
                    canvas, res = decode(actions, canvas)
                    # print('divided canvas step {}, L2Loss = {}'.format(i, ((canvas - patch_img) ** 2).mean()))
                    self.resultView.appendPlainText(
                        'divided canvas step {}, L2Loss = {}'.format(i, ((canvas - patch_img) ** 2).mean()))
                    QApplication.processEvents()
                    for j in range(5):
                        save_img(res[j], args.imgid, True)
                        args.imgid += 1

        self.resultView.appendPlainText('绘制完成！')
        assert os.path.isdir('output')

    def generate_video(self):
        os.system('2.sh')
        self.resultView.appendPlainText('视频生成完成！')
        assert os.path.isfile('video.mp4')

    def play_video(self):
        cap = cv2.VideoCapture('video.mp4')

        while (cap.isOpened()):
            ret, frame = cap.read()
            cv2.imshow('{} Press \'q\' to Exit'.format(os.path.basename(self.imgName)), frame)
            if cv2.waitKey(120) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.resultView.appendPlainText('视频播放完成！')
        QApplication.processEvents()


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = Window()
    w.resize(500, 600)
    w.show()
    sys.exit(app.exec_())
