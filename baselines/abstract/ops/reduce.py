from .base import Operator


class Reduce(Operator):
    def __init__(self):
        super().__init__()
        self.type = "reduce"
