from pathlib import Path

class Singleton:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class Config(Singleton):
    def __init__(self):
        Singleton.__init__(self)
        self.PATH_TO_IMAGES_TEST = Path("workspace/kitti/images/test/image_2")
        self.PATH_TO_IMAGES = Path("workspace/kitti/images/train/image_2")
        self.PATH_TO_LABELS = Path("workspace/kitti/images/train/label_2")
        self.PATH_TO_LABEL_MAP = Path("workspace/kitti/annotations/label_map.pbtxt")
        self.PATH_TO_TFRECORDS = Path("workspace/kitti/annotations/")
        self.PATH_TO_TENSORBOARD_LOGS = Path(".")


