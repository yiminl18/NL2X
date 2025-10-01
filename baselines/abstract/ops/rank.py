from .base import Operator


class Rank(Operator):
    def __init__(self):
        super().__init__()
        self.type = "rank"
