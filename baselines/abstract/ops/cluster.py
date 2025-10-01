from .base import Operator


class Cluster(Operator):
    def __init__(self):
        super().__init__()
        self.type = "cluster"
