from .base import Operator


class Gather(Operator):
    def __init__(self):
        super().__init__()
        self.type = "gather"
