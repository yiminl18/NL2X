from .base import Operator


class Extract(Operator):
    def __init__(self):
        super().__init__()
        self.type = "extract"
