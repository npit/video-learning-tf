class Fusion:
    """
    Aggregates the input tensor
    """
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def aggregate(self, tensor, axis):
        pass

class Avg(Fusion):
    def __init__(self):
        Fusion.__init__("Avg","Arithmetic mean fusion")

    def aggregate(self, tensor, axis):
        pass


