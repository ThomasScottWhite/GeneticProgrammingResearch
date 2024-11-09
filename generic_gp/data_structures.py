from typing import Callable, Union
from typing import Optional, Union



class Node:
    def __init__(self, value: Union[int, float, Callable], left: Optional['Node'] = None, right: Optional['Node'] = None):
        self.value = value
        self.left = left
        self.right = right

class Variable:
    def __init__(self, label: str, value: float = 0):
        self.value = value
        self.label = label
