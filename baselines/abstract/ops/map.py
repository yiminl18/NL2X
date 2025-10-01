from .base import Operator


class Map(Operator):
    def __init__(self):
        super().__init__()
        self.type = "map"
