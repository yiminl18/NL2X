from .base import Operator


class Sample(Operator):
    def __init__(self):
        super().__init__()
        self.type = "sample"
