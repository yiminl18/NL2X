from .base import Operator


class Resolve(Operator):
    def __init__(self):
        super().__init__()
        self.type = "resolve"
