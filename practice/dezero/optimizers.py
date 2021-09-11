class Optimizer:
    """
    Base Optimizer class
    """
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def udpate(self):
        params = [p for p in self.target.params() if p.grad.data is not None]

        for f in self.hooks:
            f(params)

        for p in params:
            self.update_one(p)

    def update_one(self, p):
        raise NotImplementedError()

    def add_hook(self, hook):
        self.hooks.append(hook)
