import random
import math
import matplotlib.pyplot as plt
import networkx as nx

class Individual():
    def __init__(self, tree):
        self.tree = tree
        self.fitness = None

    def evaluate(self, fitness_function, inputs, outputs):
        self.fitness = evaluate_tree(self.tree, fitness_function, inputs, outputs)

class Atom_Value():
    def __init__(self, value):
        self.value = value

class function_info():
    registered_functions = []

    def __init__(self, function, min_inputs, max_inputs, num_outputs):
        self.input_range = range(min_inputs, max_inputs + 1) 
        self.num_outputs = num_outputs
        self.function = function
        function_info.registered_functions.append(self)

    @staticmethod
    def get_registered_functions():
        return function_info.registered_functions

def register_function(min_inputs, max_inputs, num_outputs):
    def decorator(func):
        function_info(func, min_inputs, max_inputs, num_outputs)
        return func
    return decorator

class Atom_Functional():
    def __init__(self, function: function_info):
        self.function = function
        self.inputs = []

    def execute(self, inputs):
        if len(inputs) not in self.function.input_range:
            raise ValueError("Incorrect number of inputs")

        return function(inputs)
    
    def append_input(self, input):
        if len(self.inputs) == max(self.function.input_range):
            raise ValueError("Too many inputs")
        self.inputs.append(input)

    def replace_input(self, index, input):
        if len(self.inputs) == 0:
            self.append_input(input)
        else:
            self.inputs[index] = input

    def replace_rand_input(self, input):
        if len(self.inputs) == 0:
            self.append_input(input)
        else:
            index = random.choice(range(len(self.inputs)))
            self.inputs[index] = input

class Atom_Variable():

    def __init__(self, label, min_value=-float('-inf'), max_value=float('inf')):
        self.label = label
        self.min_value = min_value
        self.max_value = max_value
        self.value = None

    def set_value(self, value):
        if value < self.min_value:
            self.value = self.min_value
        elif value > self.max_value:
            self.value = self.max_value
        else:
            self.value = value
        

    def get_value(self):
        return self.value

class function_set():
    def __init__(self, functions):
        self.functions = functions

    def get_functions(self):
        return self.functions

class fitness_function():
    def __init__(self, function, inputs, outputs):
        self.function = function

    def get_function(self):
        return self.function, self.inputs, self.outputs

def copy_tree(tree):
    if isinstance(tree, Atom_Value):
        return Atom_Value(tree.value)
    elif isinstance(tree, Atom_Functional):
        atom_functional = Atom_Functional(tree.function)
        for input in tree.inputs:
            atom_functional.append_input(copy_tree(input))
        return atom_functional

def copy_individual(individual):
    individual_copy = Individual(copy_tree(individual.tree))
    individual_copy.fitness = individual.fitness
    return individual_copy
# Functional Sets

@register_function(2, 3, 1)
def add(*args):
    return sum(args)

@register_function(2, 2, 1)
def sub(*args):
    return args[0] - args[1]

@register_function(2, 2, 1)
def mul(*args):
    return args[0] * args[1]

@register_function(2, 2, 1)
def div(*args):
    if args[1] == 0:
        return args[0] / 0.2e-10
    return args[0] / args[1]

@register_function(1, 1, 1)
def pow(*args):
    return args[0] ** 2

@register_function(1, 1, 1)
def sqrt(*args):
    if args[0] <= 0:
        return 0
    return args[0] ** 0.5

def generate_tree(max_depth, varables):
    if max_depth == 0:
        if random.random() < 0.5:
            return random.choice(varables)
        return Atom_Value(random.randint(-10, 10))
    else:
        function = random.choice(function_info.get_registered_functions())
        atom_functional = Atom_Functional(function)
        
        num_inputs = random.choice(function.input_range)
        for _ in range(num_inputs):
            atom_functional.append_input(generate_tree(max_depth - 1, varables))
        return atom_functional

def plot_tree(tree):
    graph = nx.DiGraph()
    labels = {}
    _add_nodes_edges(tree, graph, labels)
    pos = _hierarchy_pos(graph, 0)
    nx.draw(graph, pos=pos, labels=labels, with_labels=True, arrows=False)
    plt.show()

def _add_nodes_edges(tree, graph, labels, parent=None, node_id=0):
    if tree is None:
        return node_id
    if isinstance(tree, Atom_Value):
        graph.add_node(node_id)
        labels[node_id] = tree.value
        if parent is not None:
            graph.add_edge(parent, node_id)
        return node_id + 1
    elif isinstance(tree, Atom_Functional):
        graph.add_node(node_id)
        labels[node_id] = tree.function.function.__name__
        if parent is not None:
            graph.add_edge(parent, node_id)
        current_id = node_id + 1
        for input_tree in tree.inputs:
            current_id = _add_nodes_edges(input_tree, graph, labels, node_id, current_id)
        return current_id
    elif isinstance(tree, Atom_Variable):
        graph.add_node(node_id)
        labels[node_id] = tree.label
        if parent is not None:
            graph.add_edge(parent, node_id)
        return node_id + 1
    
def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos_recursive(G, root, width, vert_gap, vert_loc, xcenter, pos={root: (xcenter, vert_loc)}, parent=None)
    return pos

def _hierarchy_pos_recursive(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos_recursive(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root)
    return pos

def print_tree(tree):
    if isinstance(tree, Atom_Value):
        print(tree.value)
    elif isinstance(tree, Atom_Functional):
        print(tree.function.function.__name__)
        for input in tree.inputs:
            print_tree(input)

def execute_tree(tree):
    if isinstance(tree, Atom_Value):
        return tree.value
    if isinstance(tree, Atom_Variable):
        return tree.get_value()
    elif isinstance(tree, Atom_Functional):
        inputs = [execute_tree(input) for input in tree.inputs]
        return tree.function.function(*inputs)

def mean_squared_error(result, expected):
    return (result - expected) ** 2

def evaluate_tree(tree, fitness_function, varables, inputs, outputs):
    fitness = 0
    for input, output in zip(inputs, outputs):
        for i in range(len(input)):
            varables[i].set_value(input[i])
        result = execute_tree(tree)
        fitness += fitness_function(result, output)
    return fitness/len(inputs)

def _get_all_nodes(tree):
    nodes = []
    if isinstance(tree, Atom_Value):
        nodes.append(tree)
    elif isinstance(tree, Atom_Functional):
        nodes.append(tree)
        for input in tree.inputs:
            nodes.extend(_get_all_nodes(input))
    return nodes

def _get_parent(tree, node):
    if isinstance(tree, Atom_Functional):
        for input in tree.inputs:
            if input == node:
                return tree
            parent = _get_parent(input, node)
            if parent is not None:
                return parent
    return None

def subtree_crossover(tree1, tree2):
    tree1_copy = copy_tree(tree1)
    tree2_copy = copy_tree(tree2)
    node1 = random.choice(_get_all_nodes(tree1_copy))
    node2 = random.choice(_get_all_nodes(tree2_copy))
    parent1 = _get_parent(tree1_copy, node1)
    parent2 = _get_parent(tree2_copy, node2)
    if parent1 is None:
        tree1_copy = node2
    else:
        parent1.replace_input(parent1.inputs.index(node1), node2)
    if parent2 is None:
        tree2_copy = node1
    else:
        parent2.replace_input(parent2.inputs.index(node2), node1)
    return tree1_copy, tree2_copy

def subtree_mutation(tree, varables):
    tree_copy = copy_tree(tree)
    node = random.choice(_get_all_nodes(tree_copy))
    parent = _get_parent(tree_copy, node)
    if parent is None:
        return generate_tree(3, varables)
    else:
        new_subtree = generate_tree(3, varables)
        parent.replace_input(parent.inputs.index(node), new_subtree)
        # Ensure the parent has the correct number of inputs
        while len(parent.inputs) < min(parent.function.input_range):
            parent.append_input(generate_tree(3, varables))
        return tree_copy

def create_population(population_size, max_depth, varables):
    population = []
    for _ in range(population_size):
        tree = generate_tree(max_depth, varables)
        individual = Individual(tree)
        population.append(individual)
    return population

def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness), max(tournament, key=lambda x: x.fitness)

def lexicase_selection(population, num_cases):
    cases = list(range(num_cases))
    while len(cases) > 0:
        case = random.choice(cases)
        population = sorted(population, key=lambda x: x.fitness[case], reverse=True)
        cases = [c for c in cases if population[0].fitness[c] == population[1].fitness[c]]
    return population[0]

def crossover(parent1, parent2):
    child1, child2 = subtree_crossover(parent1.tree, parent2.tree)
    child1 = Individual(child1)
    child2 = Individual(child2)
    return child1, child2

def mutation(individual, mutation_rate, varables):
    new_individual = copy_individual(individual)
    if random.random() < mutation_rate:
        individual.tree = subtree_mutation(individual.tree, varables)
    
    return new_individual

def evaluate_population(population, fitness_function, varables, inputs, outputs):
    for individual in population:
        individual.fitness = evaluate_tree(individual.tree, fitness_function, varables, inputs, outputs)

def genetic_algorithm(population_size, max_depth, num_generations, tournament_size, mutation_rate, fitness_function, inputs, outputs, varables):
    population = create_population(population_size, max_depth, varables)
    for generation in range(num_generations):
        new_population = []
        for _ in range(population_size // 2):
            evaluate_population(population, fitness_function, varables, inputs, outputs)
            parent1, parent2 = tournament_selection(population, tournament_size)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate, fitness_function, varables, inputs, outputs)
            child2 = mutation(child2, mutation_rate, fitness_function, varables, inputs, outputs)
            new_population.extend([child1, child2])
        population = new_population
        print(f'Generation {generation + 1} completed')
    return population

def validate_tree(tree, varables):
    if isinstance(tree, Atom_Value):
        return True
    elif isinstance(tree, Atom_Functional):
        if len(tree.inputs) not in tree.function.input_range:
            return False
        for input in tree.inputs:
            if not validate_tree(input, varables):
                return False
        return True
    elif isinstance(tree, Atom_Variable):
        return True
    
def main():
    inputs = [[x, x+1] for x in range(-10, 11)]
    outputs = [x**2 for x in range(-10, 11)]
    varables = [Atom_Variable(f'var_{x}') for x in enumerate(inputs[0])]
    fitness_function = mean_squared_error

    population = create_population(100, 3, varables)
    evaluate_population(population, fitness_function, varables, inputs, outputs)
    for individual in population:
        individual = mutation(individual, 1, varables)

    for individual in population:
        if validate_tree(individual.tree, varables) == False:
            plot_tree(individual.tree)
#     population_size = 100
#     max_depth = 3
#     
#     
#     

#     fitness_function = mean_squared_error
#     num_generations = 100
#     tournament_size = 7
#     mutation_rate = 0.1
#     population = genetic_algorithm(population_size, max_depth, num_generations, tournament_size, mutation_rate, fitness_function, inputs, outputs, varables)
#     best_individual = max(population, key=lambda x: x.fitness)
#     print(best_individual.fitness)
#     plot_tree(best_individual.tree)

if __name__ == "__main__":
    main()