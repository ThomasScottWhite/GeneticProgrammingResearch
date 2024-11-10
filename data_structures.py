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

    def __init__(self, label, value, min_value=-float('-inf'), max_value=float('inf')):
        self.label = label
        self.min_value = min_value
        self.max_value = max_value
        self.value = self.set_value(value)

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
