from .base import Operator


class Filter(Operator):
    def __init__(self):
        super().__init__()
        self.type = "filter"
