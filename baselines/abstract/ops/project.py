from .base import Operator


class Project(Operator):
    def __init__(self):
        super().__init__()
        self.type = "project"
