class BaseDetector:
    def detect(self, image):
        """
        image: a BGR image from cv2.imread or cap.read()

        return boxes, shape like [num_boxes, 4], a torch.tensor(device="cuda")
        """
        raise NotImplementedError
