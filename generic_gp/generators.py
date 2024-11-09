import random

from data_structures import Variable, Node


def generate_leaf(terminal_set, constant_set):
    set_choice = random.choice([terminal_set, constant_set])
    value = random.choice(set_choice)
    if value in constant_set:
        value = value()
    return Node(value)

def generate_node_tree(function_set, terminal_set, constant_set, desired_depth, current_depth=0):
    if current_depth == desired_depth:
        return generate_leaf(terminal_set, constant_set)
    else:
        value = random.choice(function_set)
        left = generate_node_tree(function_set, terminal_set, constant_set, desired_depth, current_depth + 1)
        right = generate_node_tree(function_set, terminal_set, constant_set, desired_depth, current_depth + 1)
        return Node(value, left, right)