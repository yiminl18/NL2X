from .base import Operator


class Join(Operator):
    def __init__(self):
        super().__init__()
        self.type = "join"
