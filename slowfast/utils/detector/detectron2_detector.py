from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from slowfast.utils import logging
from slowfast.utils.detector.base_detector import BaseDetector

__all__ = ['Detectron2Detector']

logger = logging.get_logger(__name__)


class Detectron2Detector(BaseDetector):
    def __init__(self, cfg):
        self.cfg = cfg
        self._person_class = cfg.DEMO.DETECTOR_PERSON_CLASS_ID

        # Load object detector from detectron2.
        dtron2_cfg_file = cfg.DEMO.DETECTOR_DETECTRON2_CFG
        dtron2_cfg = get_cfg()
        dtron2_cfg.merge_from_file(model_zoo.get_config_file(dtron2_cfg_file))
        dtron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        dtron2_cfg.MODEL.WEIGHTS = (
            cfg.DEMO.DETECTOR_DETECTRON2_MODEL_WEIGHTS)
        logger.info("Initialize detectron2 model.")
        self.object_predictor = DefaultPredictor(dtron2_cfg)

    def detect(self, image):
        outputs = self.object_predictor(image)
        fields = outputs["instances"]._fields
        pred_classes = fields["pred_classes"]
        selection_mask = pred_classes == self._person_class
        pred_classes = pred_classes[selection_mask]
        pred_boxes = fields["pred_boxes"].tensor[selection_mask]
        return pred_boxes
