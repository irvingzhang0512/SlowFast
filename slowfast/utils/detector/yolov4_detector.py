import numpy as np
from ctypes import (CDLL, POINTER, RTLD_GLOBAL, Structure, c_char_p, c_float,
                    c_int, c_void_p, pointer)
import cv2

from slowfast.utils.detector.base_detector import BaseDetector


class BOX(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("w", c_float), ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX), ("classes", c_int), ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)), ("objectness", c_float),
                ("sort_class", c_int), ("uc", POINTER(c_float)),
                ("points", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int), ("h", c_int), ("c", c_int),
                ("data", POINTER(c_float))]


class Yolov4Detector(BaseDetector):
    def __init__(self, cfg, original_height=544, original_width=960):
        self.cfg = cfg
        self._thres = 0.25
        self._nms = 0.45
        hasGPU = True

        self._config_path = cfg.DEMO.DETECTOR_YOLOV4_CONFIG_PATH
        self._weight_path = cfg.DEMO.DETECTOR_YOLOV4_WEIGHT_PATH
        self._num_classes = cfg.DEMO.DETECTOR_NUM_CLASSES
        self._target_classes = (cfg.DEMO.DETECTOR_PERSON_CLASS_ID, )
        self._origin_height = original_height
        self._origin_width = original_width

        lib = CDLL(cfg.DEMO.DETECTOR_YOLOV4_LIB_PATH, RTLD_GLOBAL)
        self.lib = lib

        lib.network_width.argtypes = [c_void_p]
        lib.network_width.restype = c_int
        lib.network_height.argtypes = [c_void_p]
        lib.network_height.restype = c_int

        load_net_custom = lib.load_network_custom
        load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        load_net_custom.restype = c_void_p
        self._net_main = load_net_custom(self._config_path.encode("ascii"),
                                         self._weight_path.encode("ascii"), 0,
                                         1)  # batch size = 1

        make_image = lib.make_image
        make_image.argtypes = [c_int, c_int, c_int]
        make_image.restype = IMAGE
        self._darknet_image = make_image(self.network_width(),
                                         self.network_height(), 3)

        self.copy_image_from_bytes = lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

        if hasGPU:
            set_gpu = lib.cuda_set_device
            set_gpu.argtypes = [c_int]

        self.get_network_boxes = lib.get_network_boxes
        self.get_network_boxes.argtypes = [
            c_void_p, c_int, c_int, c_float, c_float,
            POINTER(c_int), c_int,
            POINTER(c_int), c_int
        ]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.do_nms_sort = lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.predict_image = lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

    def network_width(self):
        return self.lib.network_width(self._net_main)

    def network_height(self):
        return self.lib.network_height(self._net_main)

    def _darknet_detect_image(self, hier_thresh=.5):
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self._net_main, self._darknet_image)
        letter_box = 0
        dets = self.get_network_boxes(self._net_main, self._darknet_image.w,
                                      self._darknet_image.h, self._thres,
                                      hier_thresh, None, 0, pnum, letter_box)
        num = pnum[0]
        if self._nms:
            self.do_nms_sort(dets, num, self._num_classes, self._nms)
        res = []
        for j in range(num):
            target = range(
                self._num_classes) if self._target_classes is None or len(
                    self._target_classes) == 0 else self._target_classes
            for i in target:
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((i, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_detections(dets, num)
        return res

    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,
                           (self.network_width(), self.network_height()),
                           interpolation=cv2.INTER_LINEAR)

        self.copy_image_from_bytes(self._darknet_image, image.tobytes())

        detections = self._darknet_detect_image()

        boxes_xywh = []
        for d in detections:
            boxes_xywh.append(d[2])
        boxes_xywh = np.array(boxes_xywh).astype(np.float32)

        boxes = np.array([])
        if len(boxes_xywh) > 0:
            boxes = np.zeros_like(boxes_xywh)
            boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # xmin
            boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # ymin
            boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # xmax
            boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # ymax
            boxes[:, ::2] = boxes[:, ::2] / self.network_width(
            ) * self._origin_width
            boxes[:, 1::2] = boxes[:, 1::2] / self.network_height(
            ) * self._origin_height
        return boxes
