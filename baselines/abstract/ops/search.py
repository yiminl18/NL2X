from .base import Operator


class Search(Operator):
    def __init__(self):
        super().__init__()
        self.type = "search"
