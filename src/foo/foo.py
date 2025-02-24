class Bar:
    x: int
    y: int

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __call__(self) -> int:
        return self.x * self.y


def foo():
    return 2
