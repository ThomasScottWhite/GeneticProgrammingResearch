import random
import matplotlib.pyplot as plt
import networkx as nx
from typing import Callable, List, Union
from typing import Optional, Union
import random
from helper import plot_tree, Node, Variable

def add(x: float, y: float) -> float:
    return x + y

def sub(x: float, y: float) -> float:
    return x - y

def mul(x: float, y: float) -> float:
    return x * y

def div(x: float, y: float) -> float:
    if y == 0:
        y = 0.001  
    return x / y

basic_math_functions: List[Callable[[float, float], float]] = [
    add,
    sub,
    mul,
    div,
]


def generate_terminal_set(input):
    terminal_set = []
    for i in range(len(input[0])):
        terminal_set.append(Variable("var_" + str(i + 1), 0))
    return terminal_set

def random_int() -> int:
    return random.randint(0, 10)


int_constant_set: List[Callable[[], int]] = [random_int]