from __future__ import annotations

from collections.abc import Iterable, Iterator
from numba import njit, float32, float64, int32, int64
import numpy as np
import pathlib

class OldSource:
    base_source: OldSource
    dist: dict[str, float]
    simbols: list[str]
    pmf: np.ndarray
    cmf: np.ndarray
    entropy: float
    epsilon: float
    typical_set: set[str]
    
    @property
    def alphabet(self) -> list[str]:
        return self.base_source.simbols

    def __init__(self, dist: dict[str, float], epsilon: float = 0.1, base_source: OldSource = None):
        self.base_source = self if base_source is None else base_source
        
        self.dist = dist
        
        self.simbols = [simbol for simbol in dist.keys()]
        
        self.pmf = np.array([float(probability) for probability in dist.values()])
        pmf_norm = self.pmf.sum()
        self.pmf /= pmf_norm
        for simbol in self.dist.keys():
            self.dist[simbol] = self.dist[simbol]/pmf_norm
        
        self.cmf = np.zeros(len(self.pmf))
        cmf = 0
        for i in range(len(self.pmf)):
            cmf += self.pmf[i]
            self.cmf[i] = cmf
        self.cmf /= cmf
        
        self.entropy = np.sum(-np.log2(self.pmf)*self.pmf)
        
        self.epsilon = epsilon
        self.typical_set = set()
        for simbol, probability in self.dist.items():
            if 2**(-len(self.simbols[0])*(self.base_source.entropy + self.epsilon)) < probability < 2**(-len(self.simbols[0])*(self.base_source.entropy - self.epsilon)):
                self.typical_set.add(simbol)
            
    def extend(self, n: int) -> OldSource:
        if n < 2:
            return self
        
        extended_dist = dict()
        aux_dist = dict()
        for simbol, probability in self.dist.items():
            extended_dist[simbol] = probability
            
        for i in range(n - 1):
                aux_dist = dict()
                for extended_simbol, extended_probability in extended_dist.items():
                    for simbol, probability in self.dist.items():
                        aux_dist[extended_simbol + simbol] = extended_probability*probability
                extended_dist = dict()
                for aux_simbol, aux_probability in aux_dist.items():
                    extended_dist[aux_simbol] = aux_probability
            
        return OldSource(extended_dist, base_source=self, epsilon=self.epsilon)
            
    def probability(self, simbols: str | list[str]) -> float:
        if isinstance(simbols, str):
            return self.dist[simbols]
        if isinstance(simbols, Iterable):
            return np.sum([self.dist[simbol] for simbol in simbols])

    def __repr__(self) -> str:
        print_output = "Simbol\t"
        while len(print_output) < len(self.simbols[0]) + 1:
            print_output += "\t"
        print_output += "Probability"
        for simbol, probability in self.dist.items():
            print_output += "\n"
            print_output += simbol
            print_output += "\t"
            print_output += f"{probability:.3g}"
        
        return print_output
    
    def __call__(self, n: int = 1) -> str:
        output = ""
        for i in range(n):
            shifted_cmf = self.cmf - np.random.random()
            positive_shifted_cmf = np.where(shifted_cmf < 0, np.inf, shifted_cmf)
            output += self.simbols[np.argmin(positive_shifted_cmf)]
            
        return output
    
    def __len__(self) -> int:
        return len(self.dist)

class RandomVariable:
    values: tuple[np.ndarray]
    pmf: np.ndarray
    dimension: int
    
    cmf: np.ndarray
    
    def __init__(self, *, random_variable: RandomVariable | float = None, pmf: np.ndarray = None, values: tuple[Iterable[float]] | Iterable[float] = None):
        if random_variable is not None:
            if isinstance(random_variable, RandomVariable):
                self.values = tuple((value.copy() for value in random_variable.values))
                self.pmf = random_variable.pmf.copy()
                self.dimension = random_variable.dimension
            else:
                try:
                    random_variable = float(random_variable)
                except (TypeError, ValueError) as e:
                    raise e(f"{type(random_variable)} is not valid type for {type(self).__name__} creation")
                else:
                    self.values = (np.array([random_variable]), )
                    self.pmf = np.array([1])
                    self.dimension = 1
        elif pmf is not None:
            self.dimension = len(pmf.shape)
            # Si solo recibe una pmf, le asigna valores enteros desde 0 en adelante
            if values is None:
                self.values = tuple((np.array([j for j in range(pmf.shape[i])]) for i in range(self.dimension)))
            else:
                if any((isinstance(value, Iterable) for value in values)):
                    assert self.dimension == len(values)
                    self.values = tuple((np.array(value if isinstance(value, Iterable) else [value]) for value in values))
                    assert all((len(self.values[i]) == pmf.shape[i] for i in range(self.dimension)))
                else:
                    assert self.dimension == 1
                    self.values = np.array(values if isinstance(values, Iterable) else [values])
                    assert len(self.values) == len(pmf)
            if self.dimension == 1:
                values_tuple_list = [(value, ) for value in values]
                values_tuple_index_list = [(i, ) for i in range(len(values))]
            else:
                values_tuple_list = [(value, ) for value in values[0]]
                values_tuple_index_list = [(i, ) for i in range(len(values[0]))]
                for i in range(1, len(values)):
                    aux_values_tuple_list = values_tuple_list
                    aux_values_tuple_index_list = values_tuple_index_list
                    values_tuple_list = []
                    values_tuple_index_list = []
                    for j in range(len(values[i])):
                        for k in range(len(aux_values_tuple_list)):
                            values_tuple_list.append((*aux_values_tuple_list[k], values[i][j]))
                            values_tuple_index_list.append((*aux_values_tuple_index_list[k], j))
            dist_dict = dict()
            for value_tuple in values_tuple_list:
                dist_dict[value_tuple] = 0
            for value_tuple, value_tuple_index in zip(values_tuple_list, values_tuple_index_list):
                dist_dict[value_tuple] += pmf[*value_tuple_index]
            # TODO: values and pmf reduction, pmf normalization, cmf
            self.values = np.array([value for value in dist_dict.keys()])
            self.pmf = np.array([probability for probability in dist_dict.values()])
            pmf_norm = self.pmf.sum()
            self.pmf /= pmf_norm
        else:
            raise TypeError("Missing 1 of 2 required keyword-only argument: 'random_variable' or 'pmf'")
            
        assert all(self.pmf >= 0)
        
        self.cmf = np.zeros(len(self.pmf))
        cmf = 0
        for i in range(len(self.pmf)):
            cmf += self.pmf[i]
            self.cmf[i] = cmf
        self.cmf /= cmf
    
    def __neg__(self) -> RandomVariable:
        return RandomVariable((-self.values, self.pmf))
            
    def __add__(self, va: RandomVariable | float) -> RandomVariable:
        if isinstance(va, RandomVariable):
            pass
        else:
            try:
                va = float(va)
            except (TypeError, ValueError) as e:
                raise e(f"Value added is not a {type(self)} object nor is convertable to float")
            return RandomVariable((self.values + va, self.pmf))
    
    def __radd__(self, va: RandomVariable | float) -> RandomVariable:
        return self + va
        
    def __sub__(self, va: RandomVariable | float) -> RandomVariable:
        return self + -va
    
    def __rsub__(self, va: RandomVariable | float) -> RandomVariable:
        return -(self - va)
    
    def __mul__(self, va: RandomVariable | float) -> RandomVariable:
        if isinstance(va, RandomVariable):
            pass
        else:
            try:
                va = float(va)
            except (TypeError, ValueError) as e:
                raise e(f"Value multiplied is not a {type(self)} object nor is convertable to float")
            return RandomVariable((self.values*va, self.pmf))
    
    def __rmul__(self, va: RandomVariable | float) -> RandomVariable:
        return self*va
    
    def __truediv__(self, va: RandomVariable | float) -> RandomVariable:
        return self*(1/va)
        
    def __rtruediv__(self, va: RandomVariable | float) -> RandomVariable:
        if va == 1:
            return RandomVariable((1/self.values, self.pmf))
        else:
            return va*(1/self)
    
    def __pow__(self, va: RandomVariable | float) -> RandomVariable:
        if isinstance(va, RandomVariable):
            pass
        else:
            try:
                va = float(va)
            except (TypeError, ValueError) as e:
                raise e(f"Value in power is not a {type(self)} object nor is convertable to float")
            return RandomVariable((self.values**va, self.pmf))

    def __rpow__(self, va: RandomVariable | float) -> RandomVariable:
        if isinstance(va, RandomVariable):
            pass
        else:
            try:
                va = float(va)
            except (TypeError, ValueError) as e:
                raise e(f"Value in power is not a {type(self)} object nor is convertable to float")
            return RandomVariable((va**self.values, self.pmf))
    
    @staticmethod
    def log(rv: RandomVariable, base: float = 2) -> RandomVariable:
        return RandomVariable((np.emath.logn(base, rv.values), rv.pmf))

    def __eq__(self, va: RandomVariable) -> bool:
        va_eq = va if isinstance(va, RandomVariable) else RandomVariable(va)
        return all(self.values == va_eq.values) and all(self.pmf == va_eq.pmf)
    
    def __len__(self) -> int:
        return len(self.values)

    def __format__(self, format_spec) -> str:
        pass
    
    def __iter__(self) -> Iterator:
        return iter(zip(self.values, self.pmf))
    
    def __containts__(self, value: float) -> bool:
        return value in self.values

    def __call__(self) -> float:
        shifted_cmf = self.cmf - np.random.random()
        positive_shifted_cmf = np.where(shifted_cmf < 0, np.inf, shifted_cmf)
        output = self.values[np.argmin(positive_shifted_cmf)]
            
        return output

    def __repr__(self) -> str:
        print_output = "Value\t"
        print_output += "Probability"
        n_lines = len(self)
        max_print = 40
        if n_lines < max_print:
            for i in range(n_lines):
                print_output += "\n"
                print_output += f"{self.values[i]:.3g}"
                print_output += "\t"
                print_output += f"{self.pmf[i]:.3g}"
        else:
            for i in range(max_print//2):
                print_output += "\n"
                print_output += f"{self.values[i]:.3g}"
                print_output += "\t"
                print_output += f"{self.pmf[i]:.3g}"
            print_output += "\n ------------"
            for i in range(n_lines - max_print//2, n_lines):
                print_output += "\n"
                print_output += f"{self.values[i]:.3g}"
                print_output += "\t"
                print_output += f"{self.pmf[i]:.3g}"
        
        return print_output

class JointRandomVariable(RandomVariable):
    values: tuple[np.ndarray]
    pmf: np.ndarray
    
    cmf: np.ndarray
    
    def __init__(self, dist: tuple[tuple[Iterable[float]], Iterable[float]] | JointRandomVariable):
        if isinstance(dist, tuple):
            dist_dict = dict()
            for value in dist[0]:
                dist_dict[value] = 0
            for value, probability in zip(dist[0], dist[1]):
                dist_dict[value] += probability
            self.values = np.array([value for value in dist_dict.keys()])
            self.pmf = np.array([probability for probability in dist_dict.values()])
            pmf_norm = self.pmf.sum()
            self.pmf /= pmf_norm
        elif isinstance(dist, RandomVariable):
            self.values = dist.values.copy()
            self.pmf = dist.pmf.copy()
        else:
            raise TypeError(f"{dist} is not valid type for {type(self).__name__} creation")
            
        assert len(self.values) == len(np.unique(self.values))
        assert all(self.pmf >= 0)
        
        self.cmf = np.zeros(len(self.pmf))
        cmf = 0
        for i in range(len(self.pmf)):
            cmf += self.pmf[i]
            self.cmf[i] = cmf
        self.cmf /= cmf

@njit(
    [float32(float32[:], int64[:], int32, int32), float64(float64[:], int64[:], int32, int32)],
    parallel=True,
    cache=True,
    )
def fast_probability(pmf: np.ndarray, simbol_indexes: np.ndarray, n_base_simbols: int, n_extension: int) -> float:
    probability = np.ones(len(simbol_indexes))
    for i in range(len(simbol_indexes)):
        var_index = simbol_indexes[i]
        for j in range(n_extension - 1, -1, -1):
            simbol_index = var_index // (n_base_simbols**j)
            probability[i] *= pmf[simbol_index]
            var_index -= simbol_index*(n_base_simbols**j)
    return np.sum(probability)

@njit(
    [float32(float32[:], int32, int32, int32), float64(float64[:], int32, int32, int32)],
    # parallel=True,
    cache=True,
    )
def fast_entropy(pmf: np.ndarray, n_base_simbols: int, n_extension: int, base: int) -> float:
    entropy_value = 0
    # extension_pmf = np.zeros(alphabet_len**n_extension)
    for i in range(n_base_simbols**n_extension):
        probability = 1
        var_index = i
        for j in range(n_extension - 1, -1, -1):
            simbol_index = var_index // (n_base_simbols**j)
            probability *= pmf[simbol_index]
            var_index -= simbol_index*(n_base_simbols**j)
        # extension_pmf[i] = probability if probability else 0
        entropy_value += -np.log2(probability)*probability if probability else 0
    # return np.sum(-np.log2(extension_pmf)*extension_pmf)
    
    return entropy_value/np.log2(base)

@njit(
    [int64[:](float32[:], float32, float32, int32, int32), int64[:](float64[:], float32, float32, int32, int32)],
    cache=True,
    )
def fast_typical_set(pmf: np.ndarray, base_entropy: float, epsilon: float, n_base_simbols: int, n_extension: int) -> np.ndarray:
    # indexes = np.zeros(alphabet_len**n_extension, int32)
    indexes = []
    for i in range(n_base_simbols**n_extension):
        probability = 1
        var_index = i
        for j in range(n_extension - 1, -1, -1):
            simbol_index = var_index // (n_base_simbols**j)
            probability *= pmf[simbol_index]
            var_index -= simbol_index*(n_base_simbols**j)
        if 2**(-n_extension*(base_entropy + epsilon)) < probability < 2**(-n_extension*(base_entropy - epsilon)):
            # indexes[i] = 1
            indexes.append(i)
    return np.array(indexes, int64)

class Source:
    dist: dict[str, float]
    alphabet: list[str]
    pmf: np.ndarray
    n_extension: int
    
    n_base_simbols: int
    n_simbols: int
    simbol_length: int
    
    cmf: np.ndarray
    base_entropy: float
    entropy_per_simbol: float
        
    def __init__(self, dist: dict[str, float] | tuple[Iterable[str], Iterable[float]] | Source, n_extension: int = 1):
        if isinstance(dist, dict):
            self.dist = {simbol: probability for simbol, probability in dist.items()}
        if isinstance(dist, tuple):
            self.dist = {simbol: probability for simbol, probability in zip(dist[0], dist[1])}
        if isinstance(dist, Source):
            self.dist = {simbol: probability for simbol, probability in dist.dist.items()}
        self.alphabet = [simbol for simbol in self.dist.keys()]
        self.n_extension = n_extension
        
        self.n_base_simbols = len(self.alphabet)
        self.n_simbols = self.n_base_simbols**self.n_extension
        self.simbol_length = len(self.alphabet[0])
        
        self.pmf = np.array([float(probability) for probability in self.dist.values()])
        pmf_norm = self.pmf.sum()
        self.pmf /= pmf_norm
        
        self.cmf = np.zeros(len(self.pmf))
        cmf = 0
        for i in range(len(self.pmf)):
            cmf += self.pmf[i]
            self.cmf[i] = cmf
        self.cmf /= cmf
        
        masked_pmf = np.ma.masked_values(self.pmf, 0)
        self.base_entropy = np.sum(-np.ma.log2(masked_pmf)*masked_pmf)
        self.entropy_per_simbol = self.base_entropy/self.simbol_length
                        
    def index_to_simbol(self, index: int) -> str:
        var_index = index
        simbol = ""
        for i in range(self.n_extension - 1, -1, -1):
            simbol_index = var_index // (self.n_base_simbols**i)
            simbol += self.alphabet[simbol_index]
            var_index -= simbol_index*(self.n_base_simbols**i)
        return simbol
    
    def simbol_to_index(self, simbol: str) -> int:
        index = 0
        for i in range(self.n_extension):
            for j in range(self.n_base_simbols):
                if simbol[i*self.simbol_length:(i + 1)*self.simbol_length] == self.alphabet[j]:
                    index += j*(self.n_base_simbols**(self.n_extension - i - 1))
        return index
    
    def probability(self, simbols: str | int | Iterable[str] | Iterable[int]) -> float:
        if isinstance(simbols, str):
            return self.probability([self.simbol_to_index(simbols)])
        if isinstance(simbols, int):
            return self.probability([simbols])
        if isinstance(simbols, Iterable):
            if isinstance(simbols, set):
                if len(simbols) == 0:
                    return 0
                first_simbol = simbols.pop()
                simbol_type = type(first_simbol)
                simbols.add(first_simbol)
            else:
                simbol_type = type(simbols[0])
            if simbol_type is str:
                return self.probability([self.simbol_to_index(simbol) for simbol in simbols])
            if simbol_type is int:
                return fast_probability(self.pmf, np.array(simbols), self.n_base_simbols, self.n_extension)
            
    def entropy(self, base: int = 2) -> float:
        return fast_entropy(self.pmf, self.n_base_simbols, self.n_extension, base)
    
    def typical_set(self, epsilon: float = 0.1) -> set[str]:
        typical_set_indexes = fast_typical_set(self.pmf, self.base_entropy, epsilon, self.n_base_simbols, self.n_extension)

        return {self.index_to_simbol(index) for index in typical_set_indexes}
    
    def __repr__(self) -> str:
        print_output = "Simbol\t"
        print_output_len = len(print_output) + 5
        while print_output_len < self.simbol_length*self.n_extension + 1:
            print_output += "\t"
            print_output_len += 6
        print_output += "Probability"
        n_lines = self.n_simbols
        max_print = 10
        if n_lines < max_print:
            for i in range(n_lines):
                simbol = self.index_to_simbol(i)
                print_output += "\n"
                print_output += simbol
                print_output += "\t"
                print_output += f"{self.probability(simbol):.3g}"
        else:
            for i in range(max_print//2):
                simbol = self.index_to_simbol(i)
                print_output += "\n"
                print_output += simbol
                print_output += "\t"
                print_output += f"{self.probability(simbol):.3g}"
            print_output += "\n ------------"
            for i in range(n_lines - max_print//2, n_lines):
                simbol = self.index_to_simbol(i)
                print_output += "\n"
                print_output += simbol
                print_output += "\t"
                print_output += f"{self.probability(simbol):.3g}"
        
        return print_output
    
    def __call__(self, n_generated: int = 1) -> str:
        output = ""
        for i in range(n_generated):
            for j in range(self.n_extension):
                shifted_cmf = self.cmf - np.random.random()
                positive_shifted_cmf = np.where(shifted_cmf < 0, np.inf, shifted_cmf)
                output += self.alphabet[np.argmin(positive_shifted_cmf)]
            
        return output
    
    def __len__(self) -> int:
        return self.n_simbols

UNKNOWN_STATE_INDEX = -1

@njit(
    [float32(float32[:, :], float32[:], int32, int64[:], int32, int32, int32), float64(float64[:, :], float64[:], int32, int64[:], int32, int32, int32)],
    parallel=True,
    cache=True,
    )
def fast_probability_memory(pmf: np.ndarray, state_pmf: np.ndarray, initial_state_index: int, simbol_indexes: np.ndarray, n_base_simbols: int, n_extension: int, memory: int) -> float:
    probability = np.ones(len(simbol_indexes))
    for i in range(len(simbol_indexes)):
        
        if initial_state_index == UNKNOWN_STATE_INDEX:
            probability[i] = 0
            for k in range(n_base_simbols**memory):
                variable_simbol_index = simbol_indexes[i]
                current_state_index = k
                conditional_probability = 1
                
                for j in range(min([memory, n_extension])):
                    single_simbol_index = variable_simbol_index // (n_base_simbols**(n_extension - 1 - j))
                    conditional_probability *= pmf[current_state_index, single_simbol_index]
                    variable_simbol_index -= single_simbol_index*(n_base_simbols**(n_extension - 1 - j))

                    new_state_index = (current_state_index % (n_base_simbols**(memory - 1)))
                    new_state_index *= n_base_simbols
                    new_state_index += single_simbol_index
                    current_state_index = new_state_index
                probability[i] += conditional_probability*state_pmf[k]
            for j in range(memory, n_extension):
                single_simbol_index = variable_simbol_index // (n_base_simbols**(n_extension - 1 - j))
                probability[i] *= pmf[current_state_index, single_simbol_index]
                variable_simbol_index -= single_simbol_index*(n_base_simbols**(n_extension - 1 - j))

                new_state_index = (current_state_index % (n_base_simbols**(memory - 1)))
                new_state_index *= n_base_simbols
                new_state_index += single_simbol_index
                current_state_index = new_state_index
        else:
            variable_simbol_index = simbol_indexes[i]
            current_state_index = initial_state_index
            for j in range(n_extension):
                single_simbol_index = variable_simbol_index // (n_base_simbols**(n_extension - 1 - j))
                probability[i] *= pmf[current_state_index, single_simbol_index]
                variable_simbol_index -= single_simbol_index*(n_base_simbols**(n_extension - 1 - j))

                new_state_index = (current_state_index % (n_base_simbols**(memory - 1)))
                new_state_index *= n_base_simbols
                new_state_index += single_simbol_index
                current_state_index = new_state_index
        
    return np.sum(probability)

@njit(
    [float64[:](float32[:, :], float32[:], int32, int32, int32), float64[:](float64[:, :], float64[:], int32, int32, int32)],
    parallel=True,
    cache=True,
    )
def fast_unconditional_pmf_memory(pmf: np.ndarray, state_pmf: np.ndarray, n_base_simbols: int, n_extension: int, memory: int) -> float:
    unconditional_pmf = np.zeros(n_base_simbols**n_extension)
    for i in range(n_base_simbols**n_extension):
        for k in range(n_base_simbols**memory):
            variable_simbol_index = i
            current_state_index = k
            conditional_probability = 1
            
            for j in range(n_extension):
                single_simbol_index = variable_simbol_index // (n_base_simbols**(n_extension - 1 - j))
                conditional_probability *= pmf[current_state_index, single_simbol_index]
                variable_simbol_index -= single_simbol_index*(n_base_simbols**(n_extension - 1 - j))

                new_state_index = (current_state_index % (n_base_simbols**(memory - 1)))
                new_state_index *= n_base_simbols
                new_state_index += single_simbol_index
                current_state_index = new_state_index
            
            unconditional_pmf[i] += conditional_probability*state_pmf[k]
                
    return unconditional_pmf

@njit(
    [int64[:](float32[:, :], float32[:], float32, float32, int32, int32, int32), int64[:](float64[:, :], float64[:], float32, float32, int32, int32, int32)],
    cache=True,
    )
def fast_typical_set_memory(pmf: np.ndarray, state_pmf: np.ndarray, base_entropy: float, epsilon: float, n_base_simbols: int, n_extension: int, memory: int) -> np.ndarray:
    # indexes = np.zeros(alphabet_len**n_extension, int32)
    indexes = []
    for i in range(n_base_simbols**n_extension):
        probability = 0
        for k in range(n_base_simbols**memory):
            variable_simbol_index = i
            current_state_index = k
            conditional_probability = 1
            
            for j in range(n_extension):
                single_simbol_index = variable_simbol_index // (n_base_simbols**(n_extension - 1 - j))
                conditional_probability *= pmf[current_state_index, single_simbol_index]
                variable_simbol_index -= single_simbol_index*(n_base_simbols**(n_extension - 1 - j))

                new_state_index = (current_state_index % (n_base_simbols**(memory - 1)))
                new_state_index *= n_base_simbols
                new_state_index += single_simbol_index
                current_state_index = new_state_index
            
            probability += conditional_probability*state_pmf[k]
            
        if 2**(-n_extension*(base_entropy + epsilon)) < probability < 2**(-n_extension*(base_entropy - epsilon)):
            # indexes[i] = 1
            indexes.append(i)
    return np.array(indexes, int64)

class MemorySource:
    alphabet: list[str]
    pmf: np.ndarray
    n_extension: int
    
    n_base_simbols: int
    n_simbols: int
    simbol_length: int
    
    n_states: int
    memory: int
    
    cmf: np.ndarray
    conditional_sources: list[Source]
    base_entropy: float
    entropy_per_simbol: float
    
    state_source: Source
    transition_matrix: np.ndarray
    state_pmf: np.ndarray
    unconditional_simple_pmf: np.ndarray
    state_cmf: np.ndarray

    current_state_index: int = UNKNOWN_STATE_INDEX

    def __init__(self, dist: tuple[Iterable[str], np.ndarray] | Source, n_extension: int = 1):
        if isinstance(dist, tuple):
            self.alphabet = [simbol for simbol in dist[0]]
            self.pmf = np.copy(dist[1])
        if isinstance(dist, Source):
            self.alphabet = [simbol for simbol in dist.alphabet]
            self.pmf = np.copy(dist.pmf)
        self.n_extension = n_extension
        assert len(self.alphabet) == self.pmf.shape[1]
        
        self.n_base_simbols = len(self.alphabet)
        self.n_simbols = self.n_base_simbols**self.n_extension
        self.simbol_length = len(self.alphabet[0])
        assert all([len(simbol) == self.simbol_length for simbol in self.alphabet])
        
        self.n_states = self.pmf.shape[0]
        self.memory = np.emath.logn(self.n_base_simbols, self.n_states)
        assert self.memory == int(self.memory)
        self.memory = int(self.memory)
        self.memory_extension_ratio = int(np.ceil(self.memory/self.n_extension))

        distributions = [{self.alphabet[i]: self.pmf[j, i] for i in range(self.n_base_simbols)} for j in range(self.n_states)]
        self.conditional_sources = [Source(dist, self.n_extension) for dist in distributions]
        
        self.pmf = np.zeros(self.pmf.shape)
        self.cmf = np.zeros(self.pmf.shape)
        for i in range(self.n_states):
            self.pmf[i, :] = self.conditional_sources[i].pmf
            cmf = 0
            for j in range(self.n_base_simbols):
                cmf += self.pmf[i, j]
                self.cmf[i, j] = cmf
            self.cmf[i, :] /= cmf
        
        self.state_pmf = np.zeros(self.n_states)
        self.state_source = Source(distributions[0], self.memory)
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_simbols):
                new_state_index = self.new_state_index(j, i)
                self.transition_matrix[i, new_state_index] += self.probability(j, i)
        values, vectors = np.linalg.eig(self.transition_matrix.transpose())
        eigenvector_index = np.argmin(abs(values - 1))
        eigenvector = np.abs(vectors[:, eigenvector_index])
        self.state_pmf = eigenvector/np.sum(eigenvector)
        self.unconditional_simple_pmf = np.matmul(self.state_pmf, self.pmf)
        self.state_cmf = np.zeros(self.n_states)
        state_cmf = 0
        for i in range(self.n_states):
            state_cmf += self.state_pmf[i]
            self.state_cmf[i] = state_cmf
        self.state_cmf /= state_cmf
        
        self.base_entropy = 0
        for state_probability, conditional_source in zip(self.state_pmf, self.conditional_sources):
            self.base_entropy += state_probability*conditional_source.base_entropy
        self.entropy_per_simbol = self.base_entropy/self.simbol_length
        
    def index_to_simbol(self, index: int) -> str:
        var_index = index
        simbol = ""
        for i in range(self.n_extension - 1, -1, -1):
            simbol_index = var_index // (self.n_base_simbols**i)
            simbol += self.alphabet[simbol_index]
            var_index -= simbol_index*(self.n_base_simbols**i)
        return simbol
        
    def simbol_to_index(self, simbol: str) -> int:
        index = 0
        for i in range(self.n_extension):
            for j in range(self.n_base_simbols):
                if simbol[i*self.simbol_length:(i + 1)*self.simbol_length] == self.alphabet[j]:
                    index += j*(self.n_base_simbols**(self.n_extension - i - 1))
        return index
        
    def index_to_state(self, index: int) -> str:
        return self.state_source.index_to_simbol(index)
        
    def state_to_index(self, simbol: str) -> int:
        return self.state_source.simbol_to_index(simbol)

    def new_state_index(self, simbol: str | int, state: str | int) -> int:
        simbol_index = simbol if isinstance(simbol, int) else self.simbol_to_index(simbol)
        state_index = state if isinstance(state, int) else self.state_to_index(state)
        
        if self.n_states > self.n_simbols:
            new_state_index = (state_index % (self.n_base_simbols**(self.memory - self.n_extension)))
            new_state_index *= self.n_base_simbols**self.n_extension
            new_state_index += simbol_index
        else:
            new_state_index = simbol_index % self.n_states
        
        return new_state_index
        
    def probability(self, simbols: str | int | Iterable[str] | Iterable[int], state: str | int = None) -> float:
        state_index = self.state_to_index(state) if isinstance(state, str) else state
        if isinstance(simbols, str):
            return self.probability([self.simbol_to_index(simbols)], state)
        if isinstance(simbols, int):
            return self.probability([simbols], state)
        if isinstance(simbols, Iterable):
            if isinstance(simbols, set):
                if len(simbols) == 0:
                    return 0
                first_simbol = simbols.pop()
                simbol_type = type(first_simbol)
                simbols.add(first_simbol)
            else:
                simbol_type = type(simbols[0])
            if simbol_type is str:
                return self.probability([self.simbol_to_index(simbol) for simbol in simbols], state)
            if simbol_type is int:
                simbols_indexes = simbols
                initial_state_index = UNKNOWN_STATE_INDEX if state is None else state_index
                return fast_probability_memory(self.pmf, self.state_pmf, initial_state_index, np.array(simbols_indexes), self.n_base_simbols, self.n_extension, self.memory)
        
    def entropy(self, base: int = 2) -> float:
        entropy = 0
        for state_probability, conditional_source in zip(self.state_pmf, self.conditional_sources):
            entropy += state_probability*conditional_source.entropy(base)
            
        return entropy
    
    def unconditional_source(self) -> Source:
        unconditional_pmf = fast_unconditional_pmf_memory(self.pmf, self.state_pmf, self.n_base_simbols, self.n_extension, self.memory)
        unconditional_alphabet = [self.index_to_simbol(i) for i in range(self.n_simbols)]
        unconditional_dist = {simbol: probability for simbol, probability in zip(unconditional_alphabet, unconditional_pmf)}

        return Source(unconditional_dist)
    
    def typical_set(self, epsilon: float = 0.1) -> set[str]:
        typical_set_indexes = fast_typical_set_memory(self.pmf, self.state_pmf, self.base_entropy, epsilon, self.n_base_simbols, self.n_extension, self.memory)

        return {self.index_to_simbol(index) for index in typical_set_indexes}
    
    def from_input(cls, input: str | pathlib.Path, encoding: str = None) -> Source:
        raise NotImplementedError
    
    def __repr__(self) -> str:
        print_output = "Simbol\t"
        print_output_len = len(print_output) + 5
        while print_output_len < self.n_extension + 1 + self.memory + 1:
            print_output += "\t"
            print_output_len += 6
        print_output += "Probability"
        n_lines = self.n_states*self.n_simbols
        max_print = 40
        if n_lines < max_print:
            for i in range(n_lines//self.n_states):
                for j in range(self.n_states):
                    simbol = self.index_to_simbol(i)
                    state = self.index_to_state(j)
                    print_output += "\n"
                    print_output += simbol
                    print_output += "|"
                    print_output += state
                    print_output += "\t"
                    print_output += f"{self.probability(simbol, state):.3g}"
        else:
            for i in range((max_print//2)//self.n_states):
                for j in range(self.n_states):
                    simbol = self.index_to_simbol(i)
                    state = self.index_to_state(j)
                    print_output += "\n"
                    print_output += simbol
                    print_output += "|"
                    print_output += state
                    print_output += "\t"
                    print_output += f"{self.probability(simbol, state):.3g}"
            print_output += "\n ------------"
            for i in range((n_lines - (max_print//2))//self.n_states, n_lines//self.n_states):
                for j in range(self.n_states):
                    simbol = self.index_to_simbol(i)
                    state = self.index_to_state(j)
                    print_output += "\n"
                    print_output += simbol
                    print_output += "|"
                    print_output += state
                    print_output += "\t"
                    print_output += f"{self.probability(simbol, state):.3g}"
        
        return print_output
    
    def __call__(self, n_generated: int = 1, state: str | int = None, reset_state: bool = False) -> str:
        output = ""
        state_index = self.state_to_index(state) if isinstance(state, str) else state
        
        if reset_state:
            self.current_state_index = UNKNOWN_STATE_INDEX
        if state_index is None:
            if self.current_state_index == UNKNOWN_STATE_INDEX:
                shifted_state_cmf = self.state_cmf - np.random.random()
                positive_shifted_state_cmf = np.where(shifted_state_cmf < 0, np.inf, shifted_state_cmf)
                state_index = int(np.argmin(positive_shifted_state_cmf))
            else:
                state_index = self.current_state_index

        for i in range(n_generated*self.n_extension):
            shifted_cmf = self.cmf[state_index] - np.random.random()
            positive_shifted_cmf = np.where(shifted_cmf < 0, np.inf, shifted_cmf)
            simbol_index = int(np.argmin(positive_shifted_cmf))
            output += self.alphabet[simbol_index]
            state_index = self.new_state_index(simbol_index, state_index)
        self.current_state_index = state_index
            
        return output
    
    def __len__(self) -> int:
        return self.n_simbols
