from .base import Operator


class Unnest(Operator):
    def __init__(self):
        super().__init__()
        self.type = "unnest"
