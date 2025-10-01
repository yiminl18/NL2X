from .base import Operator


class TopK(Operator):
    def __init__(self):
        super().__init__()
        self.type = "topk"
