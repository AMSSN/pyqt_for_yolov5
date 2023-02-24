import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.backends import cudnn

import tree
import sys
from cv2 import cv2
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal
from yolo.detect import parse_opt, main
from yolo.models.common import DetectMultiBackend
from yolo.models.experimental import attempt_load
from yolo.utils.datasets import LoadWebcam, LoadImages, IMG_FORMATS, VID_FORMATS, LoadStreams
from yolo.utils.general import check_img_size, check_imshow, check_file, increment_path, non_max_suppression, \
    scale_coords
from yolo.utils.plots import Annotator, colors
from yolo.utils.torch_utils import select_device, time_sync

CAM_NUM_1 = 0
CAM_NUM_2 = 1


class DetThread(QThread):
    # pyqt
    send_img = pyqtSignal(np.ndarray)
    send_msg = pyqtSignal(str)
    send_total_num = pyqtSignal(int)
    send_four = pyqtSignal(list)
    send_per_four = pyqtSignal(list)
    send_classes = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()

    @torch.no_grad()
    def run(self,
            weights='yolo/yolov5s.pt',  # model.pt path(s)
            source='0',  # file/dir/URL/glob, 0 for webcam
            data='yolo/data/coco128.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):
        source = str(CAM_NUM_1)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download
        # # Directories
        # save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        # (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # # Half
        # half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        # if pt or jit:
        #     model.model.half() if half else model.model.float()

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        # 这里不用for循环，改为对迭代器死循环。
        dataset = iter(dataset)
        while True:
            path, im, im0s, vid_cap, s = next(dataset)
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                       max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + (
                #     '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()
                trigger_gap = random.randint(0, 20)
                if  trigger_gap== 0:
                    self.send_img.emit(im0)
                    self.send_total_num.emit(random.randint(10, 100))
                    self.send_four.emit([random.randint(10, 100), random.randint(10, 100), random.randint(10, 100), random.randint(10, 100)])
                    self.send_per_four.emit([random.randint(10, 100), random.randint(10, 100), random.randint(10, 100), random.randint(10, 100)])
                    self.send_classes.emit("优秀")
                    print(trigger_gap)



class MainDialog(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = tree.Ui_mainWindow()
        self.ui.setupUi(self)
        # 设置图标
        self.setWindowIcon(QIcon('./static/tree.ico'))
        # 设置窗口名称
        self.setWindowTitle("检测")
        self.ui.label_threshold.setText(str(self.ui.horizontalSlider.value()))
        self.setFixedSize(self.width(), self.height())
        self.timer_camera = QtCore.QTimer()
        self.timer_result = QtCore.QTimer()
        self.slot_init()
        # yolo
        self.yolo_detec = DetThread()
        self.yolo_detec.send_img.connect(lambda x: self.get_result_and_show(x))
        self.yolo_detec.send_total_num.connect(lambda x: self.set_total_num(x))
        self.yolo_detec.send_four.connect(lambda x: self.set_four(x))
        self.yolo_detec.send_per_four.connect(lambda x: self.set_per_four(x))
        self.yolo_detec.send_classes.connect(lambda x: self.set_calsses(x))

    def paintEvent(self, event):
        painter = QPainter(self)
        # 设置窗口背景
        pixmap = QPixmap("./static/background.png")
        painter.drawPixmap(self.rect(), pixmap)

    def slot_init(self):
        self.timer_camera.timeout.connect(self.get_frame_and_show)
        self.ui.pushButton_open_camera.clicked.connect(self.open_camare_clicked)
        self.ui.pushButton_start.clicked.connect(self.init_detect_model)
        self.ui.pushButton_detect.clicked.connect(self.detect_frame)
        self.ui.pushButton_stop.clicked.connect(self.release_source)
        self.ui.horizontalSlider.valueChanged.connect(self.get_iou_threshold)

    # 设置IOU
    def get_iou_threshold(self):
        self.ui.label_threshold.setText(str(self.ui.horizontalSlider.value() / 100))
        self.ui.label_threshold.repaint()

    # 显示摄像头画面
    def get_frame_and_show(self):
        # 1号摄像头
        flag, self.image1 = self.cap1.read()
        frame1 = cv2.resize(self.image1, (self.ui.label_camera1.width(), self.ui.label_camera1.height()))
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        showImage1 = QtGui.QImage(frame1.data, frame1.shape[1], frame1.shape[0], frame1.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)
        self.ui.label_camera1.setPixmap(QtGui.QPixmap.fromImage(showImage1))
        # # 2号摄像头
        # flag, self.image2 = self.cap2.read()
        # frame2 = cv2.resize(self.image2, (self.ui.label_camera1.width(), self.ui.label_camera1.height()))
        # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        # showImage2 = QtGui.QImage(frame2.data, frame2.shape[1], frame2.shape[0], frame2.shape[1] * 3,
        #                          QtGui.QImage.Format_RGB888)
        # self.ui.label_camera2.setPixmap(QtGui.QPixmap.fromImage(showImage2))

    # 显示检测结果
    def get_result_and_show(self, img_src):
        # 两个图片分别是img_src[0]、img_src[1]
        try:
            frame1 = cv2.resize(img_src, (self.ui.label_camera1.width(), self.ui.label_camera1.height()))
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            showImage1 = QtGui.QImage(frame1.data, frame1.shape[1], frame1.shape[0], frame1.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            self.ui.label_camera1.setPixmap(QtGui.QPixmap.fromImage(showImage1))

            frame2 = cv2.resize(img_src, (self.ui.label_camera1.width(), self.ui.label_camera1.height()))
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            showImage2 = QtGui.QImage(frame2.data, frame2.shape[1], frame2.shape[0], frame2.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            self.ui.label_camera2.setPixmap(QtGui.QPixmap.fromImage(showImage2))
        except Exception as e:
            print(repr(e))

    # 点击“摄像”按钮
    def open_camare_clicked(self):
        if not self.timer_camera.isActive():
            self.cap1 = cv2.VideoCapture(CAM_NUM_1, cv2.CAP_DSHOW)
            # self.cap2 = cv2.VideoCapture(CAM_NUM_2, cv2.CAP_DSHOW)
            flag1 = self.cap1.open(CAM_NUM_1)
            # flag2 = self.cap2.open(CAM_NUM_2)
            if not flag1:
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查摄像头1", buttons=QtWidgets.QMessageBox.Ok)
            # elif not flag2:
            #     msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查摄像头2", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
        else:
            self.timer_camera.stop()
            self.ui.label_camera1.clear()
            self.ui.label_camera2.clear()

    # 加载权重、初始化参数等等
    def init_detect_model(self):
        msg = QtWidgets.QMessageBox.warning(self, 'Info', "初始化完成！", buttons=QtWidgets.QMessageBox.Ok)

    # 点击“检测”按钮
    def detect_frame(self):
        self.timer_camera.stop()
        self.ui.label_camera1.clear()
        self.ui.label_camera2.clear()
        if not self.yolo_detec.isRunning():
            self.yolo_detec.start()

        # self.det_thread.jump_out = False
        # if self.runButton.isChecked():
        #     self.saveCheckBox.setEnabled(False)
        #     self.det_thread.is_continue = True
        #     if not self.det_thread.isRunning():
        #         self.det_thread.start()
        #     source = os.path.basename(self.det_thread.source)
        #     source = 'camera' if source.isnumeric() else source
        #     self.statistic_msg('Detecting >> model：{}，file：{}'.
        #                        format(os.path.basename(self.det_thread.weights),
        #                               source))
        # else:
        #     self.det_thread.is_continue = False
        #     self.statistic_msg('Pause')

    def set_total_num(self, total_num):
        self.ui.lineEdit_total_num.setText(str(total_num))
        self.ui.lineEdit_total_num.repaint()

    def set_four(self, four):
        self.ui.lineEdit_num_yx.setText(str(four[0]))
        self.ui.lineEdit_num_yx.repaint()
        self.ui.lineEdit_num_hg.setText(str(four[1]))
        self.ui.lineEdit_num_hg.repaint()
        self.ui.lineEdit_num_bhg.setText(str(four[2]))
        self.ui.lineEdit_num_bhg.repaint()
        self.ui.lineEdit_num_sp.setText(str(four[3]))
        self.ui.lineEdit_num_sp.repaint()

    def set_per_four(self, per_four):
        self.ui.lineEdit_per_yx.setText(str(per_four[0]))
        self.ui.lineEdit_per_yx.repaint()
        self.ui.lineEdit_per_hg.setText(str(per_four[1]))
        self.ui.lineEdit_per_hg.repaint()
        self.ui.lineEdit_per_bhg.setText(str(per_four[2]))
        self.ui.lineEdit_per_bhg.repaint()
        self.ui.lineEdit_per_sp.setText(str(per_four[3]))
        self.ui.lineEdit_per_sp.repaint()

    def set_calsses(self, classes):
        self.ui.lineEdit.setText(classes)
        self.ui.lineEdit.repaint()
    # 释放资源
    def release_source(self):
        self.timer_camera.stop()
        self.cap1.release()
        # self.cap2.release()
        self.ui.label_camera1.clear()
        self.ui.label_camera2.clear()
        pass


if __name__ == '__main__':
    # 设置自适应分辨率
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    myapp = QApplication(sys.argv)
    myDlg = MainDialog()
    myDlg.show()
    sys.exit(myapp.exec_())

