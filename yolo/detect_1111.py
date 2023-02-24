# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python detect.py --weights yolov5s.pt --source 0  # detect only need weights.pt
"""

import argparse
import os
import cv2
import sys
import time
import yaml
# import pycuda.autoinit

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

from scripts.Detect_Result_Process import DetectResultProcess
from scripts.PythonSerials import PySerial
from utils.plots import colors
from pathlib import Path
from utils.datasets import LoadCorkDiskImages, LoadMindVisionCameraStreams
from utils.general import (LOGGER, check_img_size, print_args, scale_coords)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(engines=ROOT / 'TensorRT/yolov5s.engine',  # model.pt path(s)
           data=ROOT / 'data/Cork_disk.yaml',
           source=ROOT / '0',            # file/dir/URL/glob, 0 for webcam
           imgsz=(640, 640),             # inference size (pixels)
           batch_size=2,                 # batch_size
           conf_thres=0.45,              # confidence threshold
           iou_thres=0.70,               # NMS IOU threshold
           view_img=True,                # show results
           ):
    source = str(source)
    engine = str(engines[0] if isinstance(engines, list) else engines)

    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    imgsz = [check_img_size(x, s=32) for x in imgsz]
    dt, seen = [0.0, 0.0, 0.0], 0    # dt 可能是时间差分的意思
    all_image, true_image = 0, 0

    with open(data, errors='ignore') as f:
        data_dict = yaml.safe_load(f)  # dictionary
    names = data_dict['names']  # names: ["qualified", "unqualified", "exocuticle", "black_block", "hole", "gap"]
    disk_class = data_dict['disk_class']  # disk_class: ["good", "good_bad", "bad", "exocuticle"]
    num_class = data_dict['nc']
    num_output = num_class + 5

    DRP = DetectResultProcess(data='./data/Cork_disk.yaml')   # 这个用来处理检测结果的

    # TensorRT 引擎初始化
    cuda.init()
    ctx = cuda.Device(0).make_context()
    engine = _load_engine(engine)
    context = engine.create_execution_context()
    context.set_binding_shape(0, (batch_size, 3, *imgsz))
    inputs, outputs, bindings, stream = _allocate_buffers(engine, max_batch_size=batch_size)  # 构建输入，输出，流指针

    if source.isnumeric():
        test_model = False
        view_img = True
        dataset = LoadMindVisionCameraStreams(source, img_size=imgsz, stride=32, auto=False)  # 工业相机数据流
    else:
        test_model = True
        dataset = LoadCorkDiskImages(source, img_size=imgsz, stride=32, auto=False)  # 图片流

    #  img: 用来处理的numpy矩阵，   img_orig：原始图片
    # 因为多幅图像，所以img的维度是 [2， 640， 640]
    for path, img, img_orig, vid_cap, print_str in dataset:
        if img is not None:
            seen += 1
            cork_class = [-1, -1]
            img_raws = [None] * 2
            t1 = time.time()
            if len(img.shape) == 3:
                img = img[None]

            np.copyto(inputs[0].host, img.ravel())
            t2 = time.time()
            dt[0] += t2 - t1
            ctx.push()
            results = do_inference(context, bindings, inputs, outputs, stream)
            ctx.pop()
            t3 = time.time()
            dt[1] += t3 - t2

            results = np.reshape(results, [batch_size, -1, num_output])

            print_str += f'preprocessing time: ({t2 - t1:.3f}s), inference time: ({t3 - t2:.3f}s)'

            # 因为我是同时处理2幅图像，所以检测结果也是2个，要分开处理
            for i, result in enumerate(results):   # results = [2, -1, 11] --> [-1, 11]
                # 这里将原始图片进行一次复制，就是用来在图片上画框和检测结果的，这个图片后面会保存下来。
                if source.isnumeric():
                    img_raw = img_orig[i].copy()
                else:
                    img_raw = img_orig[i].copy()

                # 滤波和NMS
                proposal_det = filter_boxes(result, conf_thres)
                det, no_cork_disk = multiclass_non_max_suppression(proposal_det, iou_thres, num_class)

                # 如果有检测结果，分析结果并画图
                if not no_cork_disk:
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_raw.shape).round()

                    det = DRP.detect_result_process(det, img_raw, img_raw.shape[:2], draw=True)

                    # 这里就是提取每一张图片的检测结果，cork_class 是一个列表，用来记录2幅图片的结果
                    det_disk = det[det[:, -1] < 3]
                    if len(det_disk) == 1:
                        cork_class[i] = det_disk[0, -1]
                    else:  # 出现这种情况取有最大分数的预测结果
                        max_ind = np.argmax(det_disk[:, 4])
                        cork_class[i] = det_disk[max_ind, -1]

                    # 画图
                    for *box, conf, cls in reversed(det):  # per image
                            x1, y1, x2, y2 = np.int32(box)
                            c = int(cls)
                            conf *= 100
                            cv2.rectangle(img_raw, (x1, y1), (x2, y2), colors(c, True), thickness=3)

                    img_raws[i] = img_raw
                    # 这里就是单幅图片的CV2显示了，到时候可以用一个 img_raws = [img_raws1， img_raws2] 将结果保存了，一起显示
                    if view_img:
                        cv2.imshow(str('camera capture %d' %i), img_raw)
                        cv2.waitKey(1)  # 1 millisecond

            # 这里是分析一下最终的结果了，cork_disk_class = [0, 1, 2, 3]
            # disk_class: ["good", "good_bad", "bad", "exocuticle"]
            if test_model:
                cork_disk_class = -1
                image_name = path.split('/')[-1].lower().split('.')[0]
                target = disk_class.index(image_name[:-10])

                cls1, cls2 = cork_class[0], cork_class[1]

                if cls1 == 0 and cls2 == 0:
                    cork_disk_class = 0
                elif (cls1 == 1 and cls2 == 0) or (cls1 == 0 and cls2 == 1):
                    cork_disk_class = 1
                elif cls1 == 1 and cls2 == 1:
                    cork_disk_class = 2
                elif cls1 == 2 or cls2 == 2:
                    cork_disk_class = 3

                all_image += 1
                if cork_disk_class == target:
                    true_image += 1
 
            t4 = time.time()
            dt[2] += t4 - t3
            print_str += f', post-procession time: ({t4 - t3:.3f}s), all time: ({t4 - t1:.3f}s), '
            LOGGER.info(f'{print_str}Done.')

    ctx.pop()
    # Print results
    print(true_image, all_image, true_image / all_image)
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {(1, 3, *imgsz)}' % t)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engines', nargs='+', type=str, default=ROOT / 'onnx/yolov5_1_trt.engine', help='model path(s)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/Cork_disk.yaml', help='dataset.yaml path')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument("--batch_size", type=int, default=2, help='batch_size')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--view-img', action='store_true', help='show results')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    detect(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
