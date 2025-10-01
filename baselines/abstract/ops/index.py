from .base import Operator


class Index(Operator):
    def __init__(self):
        super().__init__()
        self.type = "index"
