from .base import Operator


class Split(Operator):
    def __init__(self):
        super().__init__()
        self.type = "split"
