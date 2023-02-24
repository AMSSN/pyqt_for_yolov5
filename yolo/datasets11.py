# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob  # python自己带的一个文件操作相关模块，查找符合自己目的的文件(如模糊匹配)
import hashlib  # 哈希模块，提供了多种安全方便的hash方法
import json  # json文件操作模块
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool  # 多线程模块 线程池
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import platform
from scripts import mvsdk
from scripts.Mindvision_Camera import Camera
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps   # 图片、相机操作模块
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (LOGGER, check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first


class LoadMindVisionCameraStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        # 是否为文件(.txt)，如果是，则读取里面的每一行的内容；如果source = 0，则为0
        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            # sources = [sources]
            sources = [x for x in sources]

        n = len(sources)   # i.e source = 0, len = 1
        self.trigger = [False] * n
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n  # define space [0, 0]
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto

        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)

        if nDev < 1:
            print("No camera was found!")
            return

        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam

            cam = Camera(DevList[s])
            cam.open()
            frame = cam.grab()

            h, w = frame.shape[:2]
            self.fps[i], self.frames[i] = 30, float('inf')  # infinite stream fallback

            self.imgs[i] = frame  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cam]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

    def update(self, i, cam):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while n < f:
            n += 1
            if n % read == 0:
                im = cam.grab()
                self.imgs[i] = im
                if im is not None:
                    self.trigger[i] = True
            else:
                LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                self.imgs[i] *= 0
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        if (img0[0] is not None) and (img0[1] is not None) and self.trigger[0] and self.trigger[1]:
            img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]
            # img.astype(np.float16)比img.astype(np.float32)慢一倍，是因为 c 中没有对应的 float16。
            # 由于python是基于c的，numpy创建了一个方法来执行float16，先进行 int8 到 float32 的转换，然后再进行反向转换
            # 建议先使用 np.flaot32 类型，因为归一化处理速度很快, 然后再转换为半精度，这样copyto就不耗时间了
            img = np.stack(img, 0).astype(np.float32)  # 2ms
            img /= 255.0  # 8ms
            img = img.astype(np.float32)  # 18ms,在下一句代码后使用为28ms
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # 0ms, BGR to RGB, BHWC to BCHW
            img = np.ascontiguousarray(img)  # 6ms
        else:
            img = None

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


