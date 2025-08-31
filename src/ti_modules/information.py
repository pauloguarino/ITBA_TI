from __future__ import annotations

from collections.abc import Iterable, Sized
from numba import njit, float32, float64, int32, int64
import numpy as np
import pathlib

class OldSource:
    base_source: OldSource
    dist: dict[str, float]
    symbols: list[str]
    pmf: np.ndarray
    cmf: np.ndarray
    entropy: float
    epsilon: float
    typical_set: set[str]
    
    @property
    def alphabet(self) -> list[str]:
        return self.base_source.symbols

    def __init__(self, dist: dict[str, float], epsilon: float = 0.1, base_source: OldSource | None = None):
        self.base_source = self if base_source is None else base_source
        
        self.dist = dist
        
        self.symbols = [symbol for symbol in dist.keys()]
        
        self.pmf = np.array([float(probability) for probability in dist.values()])
        pmf_norm = self.pmf.sum()
        self.pmf /= pmf_norm
        for symbol in self.dist.keys():
            self.dist[symbol] = self.dist[symbol]/pmf_norm
        
        self.cmf = np.zeros(len(self.pmf))
        cmf = 0
        for i in range(len(self.pmf)):
            cmf += self.pmf[i]
            self.cmf[i] = cmf
        self.cmf /= cmf
        
        self.entropy = np.sum(-np.log2(self.pmf)*self.pmf)
        
        self.epsilon = epsilon
        self.typical_set = set()
        for symbol, probability in self.dist.items():
            if 2**(-len(self.symbols[0])*(self.base_source.entropy + self.epsilon)) < probability < 2**(-len(self.symbols[0])*(self.base_source.entropy - self.epsilon)):
                self.typical_set.add(symbol)
            
    def extend(self, n: int) -> OldSource:
        if n < 2:
            return self
        
        extended_dist = dict()
        aux_dist = dict()
        for symbol, probability in self.dist.items():
            extended_dist[symbol] = probability
            
        for i in range(n - 1):
                aux_dist = dict()
                for extended_symbol, extended_probability in extended_dist.items():
                    for symbol, probability in self.dist.items():
                        aux_dist[extended_symbol + symbol] = extended_probability*probability
                extended_dist = dict()
                for aux_symbol, aux_probability in aux_dist.items():
                    extended_dist[aux_symbol] = aux_probability
            
        return OldSource(extended_dist, base_source=self, epsilon=self.epsilon)
            
    def probability(self, symbols: str | list[str]) -> float:
        if isinstance(symbols, str):
            return self.dist[symbols]
        if isinstance(symbols, Iterable):
            return np.sum([self.dist[symbol] for symbol in symbols])

    def __repr__(self) -> str:
        print_output = "symbol\t"
        while len(print_output) < len(self.symbols[0]) + 1:
            print_output += "\t"
        print_output += "Probability"
        for symbol, probability in self.dist.items():
            print_output += "\n"
            print_output += symbol
            print_output += "\t"
            print_output += f"{probability:.3g}"
        
        return print_output
    
    def __call__(self, n: int = 1) -> str:
        output = ""
        for i in range(n):
            shifted_cmf = self.cmf - np.random.random()
            positive_shifted_cmf = np.where(shifted_cmf < 0, np.inf, shifted_cmf)
            output += self.symbols[np.argmin(positive_shifted_cmf)]
            
        return output
    
    def __len__(self) -> int:
        return len(self.dist)

@njit(
    [float32(float32[:], int64[:], int32, int32), float64(float64[:], int64[:], int32, int32)],
    parallel=True,
    cache=True,
    )
def fast_probability(pmf: np.ndarray, symbol_indexes: np.ndarray, n_base_symbols: int, n_extension: int) -> float:
    probability = np.ones(len(symbol_indexes))
    for i in range(len(symbol_indexes)):
        var_index = symbol_indexes[i]
        for j in range(n_extension - 1, -1, -1):
            symbol_index = var_index // (n_base_symbols**j)
            probability[i] *= pmf[symbol_index]
            var_index -= symbol_index*(n_base_symbols**j)
    return np.sum(probability)

@njit(
    [float32(float32[:], int32, int32, int32), float64(float64[:], int32, int32, int32)],
    # parallel=True,
    cache=True,
    )
def fast_entropy(pmf: np.ndarray, n_base_symbols: int, n_extension: int, base: int) -> float:
    entropy_value = 0
    # extension_pmf = np.zeros(alphabet_len**n_extension)
    for i in range(n_base_symbols**n_extension):
        probability = 1
        var_index = i
        for j in range(n_extension - 1, -1, -1):
            symbol_index = var_index // (n_base_symbols**j)
            probability *= pmf[symbol_index]
            var_index -= symbol_index*(n_base_symbols**j)
        # extension_pmf[i] = probability if probability else 0
        entropy_value += -np.log2(probability)*probability if probability else 0
    # return np.sum(-np.log2(extension_pmf)*extension_pmf)
    
    return entropy_value/np.log2(base)

@njit(
    [int64[:](float32[:], float32, float32, int32, int32), int64[:](float64[:], float32, float32, int32, int32)],
    cache=True,
    )
def fast_typical_set(pmf: np.ndarray, base_entropy: float, epsilon: float, n_base_symbols: int, n_extension: int) -> np.ndarray:
    # indexes = np.zeros(alphabet_len**n_extension, int32)
    indexes = []
    for i in range(n_base_symbols**n_extension):
        probability = 1
        var_index = i
        for j in range(n_extension - 1, -1, -1):
            symbol_index = var_index // (n_base_symbols**j)
            probability *= pmf[symbol_index]
            var_index -= symbol_index*(n_base_symbols**j)
        if 2**(-n_extension*(base_entropy + epsilon)) < probability < 2**(-n_extension*(base_entropy - epsilon)):
            # indexes[i] = 1
            indexes.append(i)
    return np.array(indexes, np.int64)

class Source:
    dist: dict[str, float]
    alphabet: list[str]
    pmf: np.ndarray
    n_extension: int
    
    n_base_symbols: int
    n_symbols: int
    symbol_length: int
    
    cmf: np.ndarray
    base_entropy: float
    entropy_per_symbol: float
        
    def __init__(self, dist: dict[str, float] | tuple[Iterable[str], Iterable[float]] | Source, n_extension: int = 1):
        if isinstance(dist, dict):
            self.dist = {symbol: probability for symbol, probability in dist.items()}
        if isinstance(dist, tuple):
            self.dist = {symbol: probability for symbol, probability in zip(dist[0], dist[1])}
        if isinstance(dist, Source):
            self.dist = {symbol: probability for symbol, probability in dist.dist.items()}
        self.alphabet = [symbol for symbol in self.dist.keys()]
        self.n_extension = n_extension
        
        self.n_base_symbols = len(self.alphabet)
        self.n_symbols = self.n_base_symbols**self.n_extension
        self.symbol_length = len(self.alphabet[0])
        
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
        self.entropy_per_symbol = self.base_entropy/self.symbol_length
                        
    def index_to_symbol(self, index: int) -> str:
        var_index = index
        symbol = ""
        for i in range(self.n_extension - 1, -1, -1):
            symbol_index = var_index // (self.n_base_symbols**i)
            symbol += self.alphabet[symbol_index]
            var_index -= symbol_index*(self.n_base_symbols**i)
        return symbol
    
    def symbol_to_index(self, symbol: str) -> int:
        index = 0
        for i in range(self.n_extension):
            for j in range(self.n_base_symbols):
                if symbol[i*self.symbol_length:(i + 1)*self.symbol_length] == self.alphabet[j]:
                    index += j*(self.n_base_symbols**(self.n_extension - i - 1))
        return index
    
    def probability(self, symbols: str | int | Iterable[str] | Iterable[int]) -> float:
        if isinstance(symbols, str):
            return self.probability([self.symbol_to_index(symbols)])
        if isinstance(symbols, int):
            return self.probability([symbols])
        if isinstance(symbols, Iterable):
            if isinstance(symbols, Sized):
                if len(symbols) == 0:
                    return 0
                first_symbol = next((symbol for symbol in symbols))
                if isinstance(first_symbol, str):
                    return self.probability([self.symbol_to_index(symbol) for symbol in symbols if isinstance(symbol, str)])
                if isinstance(first_symbol, int):
                    return fast_probability(self.pmf, np.array(symbols), self.n_base_symbols, self.n_extension)
                else:
                    raise TypeError
            else:
                raise TypeError
        else:
            raise TypeError
            
    def entropy(self, base: int = 2) -> float:
        return fast_entropy(self.pmf, self.n_base_symbols, self.n_extension, base)
    
    def typical_set(self, epsilon: float = 0.1) -> set[str]:
        typical_set_indexes = fast_typical_set(self.pmf, self.base_entropy, epsilon, self.n_base_symbols, self.n_extension)

        return {self.index_to_symbol(index) for index in typical_set_indexes}
    
    def __repr__(self) -> str:
        print_output = "symbol\t"
        print_output_len = len(print_output) + 5
        while print_output_len < self.symbol_length*self.n_extension + 1:
            print_output += "\t"
            print_output_len += 6
        print_output += "Probability"
        n_lines = self.n_symbols
        max_print = 10
        if n_lines < max_print:
            for i in range(n_lines):
                symbol = self.index_to_symbol(i)
                print_output += "\n"
                print_output += symbol
                print_output += "\t"
                print_output += f"{self.probability(symbol):.3g}"
        else:
            for i in range(max_print//2):
                symbol = self.index_to_symbol(i)
                print_output += "\n"
                print_output += symbol
                print_output += "\t"
                print_output += f"{self.probability(symbol):.3g}"
            print_output += "\n ------------"
            for i in range(n_lines - max_print//2, n_lines):
                symbol = self.index_to_symbol(i)
                print_output += "\n"
                print_output += symbol
                print_output += "\t"
                print_output += f"{self.probability(symbol):.3g}"
        
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
        return self.n_symbols

UNKNOWN_STATE_INDEX = -1

@njit(
    [float32(float32[:, :], float32[:], int32, int64[:], int32, int32, int32), float64(float64[:, :], float64[:], int32, int64[:], int32, int32, int32)],
    parallel=True,
    cache=True,
    )
def fast_probability_memory(pmf: np.ndarray, state_pmf: np.ndarray, initial_state_index: int, symbol_indexes: np.ndarray, n_base_symbols: int, n_extension: int, memory: int) -> float:
    probability = np.ones(len(symbol_indexes))
    for i in range(len(symbol_indexes)):
        
        if initial_state_index == UNKNOWN_STATE_INDEX:
            probability[i] = 0
            for k in range(n_base_symbols**memory):
                variable_symbol_index = symbol_indexes[i]
                current_state_index = k
                conditional_probability = 1
                
                for j in range(min([memory, n_extension])):
                    single_symbol_index = variable_symbol_index // (n_base_symbols**(n_extension - 1 - j))
                    conditional_probability *= pmf[current_state_index, single_symbol_index]
                    variable_symbol_index -= single_symbol_index*(n_base_symbols**(n_extension - 1 - j))

                    new_state_index = (current_state_index % (n_base_symbols**(memory - 1)))
                    new_state_index *= n_base_symbols
                    new_state_index += single_symbol_index
                    current_state_index = new_state_index
                probability[i] += conditional_probability*state_pmf[k]
            for j in range(memory, n_extension):
                single_symbol_index = variable_symbol_index // (n_base_symbols**(n_extension - 1 - j))
                probability[i] *= pmf[current_state_index, single_symbol_index]
                variable_symbol_index -= single_symbol_index*(n_base_symbols**(n_extension - 1 - j))

                new_state_index = (current_state_index % (n_base_symbols**(memory - 1)))
                new_state_index *= n_base_symbols
                new_state_index += single_symbol_index
                current_state_index = new_state_index
        else:
            variable_symbol_index = symbol_indexes[i]
            current_state_index = initial_state_index
            for j in range(n_extension):
                single_symbol_index = variable_symbol_index // (n_base_symbols**(n_extension - 1 - j))
                probability[i] *= pmf[current_state_index, single_symbol_index]
                variable_symbol_index -= single_symbol_index*(n_base_symbols**(n_extension - 1 - j))

                new_state_index = (current_state_index % (n_base_symbols**(memory - 1)))
                new_state_index *= n_base_symbols
                new_state_index += single_symbol_index
                current_state_index = new_state_index
        
    return np.sum(probability)

@njit(
    [float64[:](float32[:, :], float32[:], int32, int32, int32), float64[:](float64[:, :], float64[:], int32, int32, int32)],
    parallel=True,
    cache=True,
    )
def fast_unconditional_pmf_memory(pmf: np.ndarray, state_pmf: np.ndarray, n_base_symbols: int, n_extension: int, memory: int) -> np.ndarray:
    unconditional_pmf = np.zeros(n_base_symbols**n_extension)
    for i in range(n_base_symbols**n_extension):
        for k in range(n_base_symbols**memory):
            variable_symbol_index = i
            current_state_index = k
            conditional_probability = 1
            
            for j in range(n_extension):
                single_symbol_index = variable_symbol_index // (n_base_symbols**(n_extension - 1 - j))
                conditional_probability *= pmf[current_state_index, single_symbol_index]
                variable_symbol_index -= single_symbol_index*(n_base_symbols**(n_extension - 1 - j))

                new_state_index = (current_state_index % (n_base_symbols**(memory - 1)))
                new_state_index *= n_base_symbols
                new_state_index += single_symbol_index
                current_state_index = new_state_index
            
            unconditional_pmf[i] += conditional_probability*state_pmf[k]
                
    return unconditional_pmf

@njit(
    [int64[:](float32[:, :], float32[:], float32, float32, int32, int32, int32), int64[:](float64[:, :], float64[:], float32, float32, int32, int32, int32)],
    cache=True,
    )
def fast_typical_set_memory(pmf: np.ndarray, state_pmf: np.ndarray, base_entropy: float, epsilon: float, n_base_symbols: int, n_extension: int, memory: int) -> np.ndarray:
    # indexes = np.zeros(alphabet_len**n_extension, int32)
    indexes = []
    for i in range(n_base_symbols**n_extension):
        probability = 0
        for k in range(n_base_symbols**memory):
            variable_symbol_index = i
            current_state_index = k
            conditional_probability = 1
            
            for j in range(n_extension):
                single_symbol_index = variable_symbol_index // (n_base_symbols**(n_extension - 1 - j))
                conditional_probability *= pmf[current_state_index, single_symbol_index]
                variable_symbol_index -= single_symbol_index*(n_base_symbols**(n_extension - 1 - j))

                new_state_index = (current_state_index % (n_base_symbols**(memory - 1)))
                new_state_index *= n_base_symbols
                new_state_index += single_symbol_index
                current_state_index = new_state_index
            
            probability += conditional_probability*state_pmf[k]
            
        if 2**(-n_extension*(base_entropy + epsilon)) < probability < 2**(-n_extension*(base_entropy - epsilon)):
            # indexes[i] = 1
            indexes.append(i)
    return np.array(indexes, np.int64)

class MemorySource:
    alphabet: list[str]
    pmf: np.ndarray
    n_extension: int
    
    n_base_symbols: int
    n_symbols: int
    symbol_length: int
    
    n_states: int
    memory: int
    
    cmf: np.ndarray
    conditional_sources: list[Source]
    base_entropy: float
    entropy_per_symbol: float
    
    state_source: Source
    transition_matrix: np.ndarray
    state_pmf: np.ndarray
    unconditional_simple_pmf: np.ndarray
    state_cmf: np.ndarray

    current_state_index: int = UNKNOWN_STATE_INDEX

    def __init__(self, dist: tuple[Iterable[str], np.ndarray] | MemorySource, n_extension: int = 1):
        if isinstance(dist, tuple):
            self.alphabet = [symbol for symbol in dist[0]]
            self.pmf = np.copy(dist[1])
        if isinstance(dist, MemorySource):
            self.alphabet = [symbol for symbol in dist.alphabet]
            self.pmf = np.copy(dist.pmf)
        self.n_extension = n_extension
        assert len(self.alphabet) == self.pmf.shape[1]
        
        self.n_base_symbols = len(self.alphabet)
        self.n_symbols = self.n_base_symbols**self.n_extension
        self.symbol_length = len(self.alphabet[0])
        assert all([len(symbol) == self.symbol_length for symbol in self.alphabet])
        
        self.n_states = self.pmf.shape[0]
        self.memory = np.emath.logn(self.n_base_symbols, self.n_states)
        assert self.memory == int(self.memory)
        self.memory = int(self.memory)
        self.memory_extension_ratio = int(np.ceil(self.memory/self.n_extension))

        distributions = [{self.alphabet[i]: self.pmf[j, i] for i in range(self.n_base_symbols)} for j in range(self.n_states)]
        self.conditional_sources = [Source(dist, self.n_extension) for dist in distributions]
        
        self.pmf = np.zeros(self.pmf.shape)
        self.cmf = np.zeros(self.pmf.shape)
        for i in range(self.n_states):
            self.pmf[i, :] = self.conditional_sources[i].pmf
            cmf = 0
            for j in range(self.n_base_symbols):
                cmf += self.pmf[i, j]
                self.cmf[i, j] = cmf
            self.cmf[i, :] /= cmf
        
        self.state_pmf = np.zeros(self.n_states)
        self.state_source = Source(distributions[0], self.memory)
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_symbols):
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
        self.entropy_per_symbol = self.base_entropy/self.symbol_length
        
    def index_to_symbol(self, index: int) -> str:
        var_index = index
        symbol = ""
        for i in range(self.n_extension - 1, -1, -1):
            symbol_index = var_index // (self.n_base_symbols**i)
            symbol += self.alphabet[symbol_index]
            var_index -= symbol_index*(self.n_base_symbols**i)
        return symbol
        
    def symbol_to_index(self, symbol: str) -> int:
        index = 0
        for i in range(self.n_extension):
            for j in range(self.n_base_symbols):
                if symbol[i*self.symbol_length:(i + 1)*self.symbol_length] == self.alphabet[j]:
                    index += j*(self.n_base_symbols**(self.n_extension - i - 1))
        return index
        
    def index_to_state(self, index: int) -> str:
        return self.state_source.index_to_symbol(index)
        
    def state_to_index(self, symbol: str) -> int:
        return self.state_source.symbol_to_index(symbol)

    def new_state_index(self, symbol: str | int, state: str | int) -> int:
        symbol_index = symbol if isinstance(symbol, int) else self.symbol_to_index(symbol)
        state_index = state if isinstance(state, int) else self.state_to_index(state)
        
        if self.n_states > self.n_symbols:
            new_state_index = (state_index % (self.n_base_symbols**(self.memory - self.n_extension)))
            new_state_index *= self.n_base_symbols**self.n_extension
            new_state_index += symbol_index
        else:
            new_state_index = symbol_index % self.n_states
        
        return new_state_index
        
    def probability(self, symbols: str | int | Iterable[str] | Iterable[int], state: str | int | None = None) -> float:
        state_index = self.state_to_index(state) if isinstance(state, str) else state
        if isinstance(symbols, str):
            return self.probability([self.symbol_to_index(symbols)], state)
        if isinstance(symbols, int):
            return self.probability([symbols], state)
        if isinstance(symbols, Iterable):
            if isinstance(symbols, Sized):
                if len(symbols) == 0:
                    return 0
                first_symbol = next((symbol for symbol in symbols))
                if isinstance(first_symbol, str):
                    return self.probability([self.symbol_to_index(symbol) for symbol in symbols if isinstance(symbol, str)], state)
                if isinstance(first_symbol, int):
                    symbols_indexes = symbols
                    initial_state_index = UNKNOWN_STATE_INDEX if state_index is None else state_index
                    return fast_probability_memory(self.pmf, self.state_pmf, initial_state_index, np.array(symbols_indexes), self.n_base_symbols, self.n_extension, self.memory)
                else:
                    raise TypeError
            else:
                raise TypeError
        else:
            raise TypeError
        
    def entropy(self, base: int = 2) -> float:
        entropy = 0
        for state_probability, conditional_source in zip(self.state_pmf, self.conditional_sources):
            entropy += state_probability*conditional_source.entropy(base)
            
        return entropy
    
    def unconditional_source(self) -> Source:
        unconditional_pmf = fast_unconditional_pmf_memory(self.pmf, self.state_pmf, self.n_base_symbols, self.n_extension, self.memory)
        unconditional_alphabet = [self.index_to_symbol(i) for i in range(self.n_symbols)]
        unconditional_dist = {symbol: probability for symbol, probability in zip(unconditional_alphabet, unconditional_pmf)}

        return Source(unconditional_dist)
    
    def typical_set(self, epsilon: float = 0.1) -> set[str]:
        typical_set_indexes = fast_typical_set_memory(self.pmf, self.state_pmf, self.base_entropy, epsilon, self.n_base_symbols, self.n_extension, self.memory)

        return {self.index_to_symbol(index) for index in typical_set_indexes}
    
    @classmethod
    def from_input(cls, input: str | pathlib.Path, encoding: str | None = None) -> MemorySource:
        raise NotImplementedError
    
    def __repr__(self) -> str:
        print_output = "symbol\t"
        print_output_len = len(print_output) + 5
        while print_output_len < self.n_extension + 1 + self.memory + 1:
            print_output += "\t"
            print_output_len += 6
        print_output += "Probability"
        n_lines = self.n_states*self.n_symbols
        max_print = 40
        if n_lines < max_print:
            for i in range(n_lines//self.n_states):
                for j in range(self.n_states):
                    symbol = self.index_to_symbol(i)
                    state = self.index_to_state(j)
                    print_output += "\n"
                    print_output += symbol
                    print_output += "|"
                    print_output += state
                    print_output += "\t"
                    print_output += f"{self.probability(symbol, state):.3g}"
        else:
            for i in range((max_print//2)//self.n_states):
                for j in range(self.n_states):
                    symbol = self.index_to_symbol(i)
                    state = self.index_to_state(j)
                    print_output += "\n"
                    print_output += symbol
                    print_output += "|"
                    print_output += state
                    print_output += "\t"
                    print_output += f"{self.probability(symbol, state):.3g}"
            print_output += "\n ------------"
            for i in range((n_lines - (max_print//2))//self.n_states, n_lines//self.n_states):
                for j in range(self.n_states):
                    symbol = self.index_to_symbol(i)
                    state = self.index_to_state(j)
                    print_output += "\n"
                    print_output += symbol
                    print_output += "|"
                    print_output += state
                    print_output += "\t"
                    print_output += f"{self.probability(symbol, state):.3g}"
        
        return print_output
    
    def __call__(self, n_generated: int = 1, state: str | int | None = None, reset_state: bool = False) -> str:
        output = ""
        state_index = self.state_to_index(state) if isinstance(state, str) else None
        
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
            symbol_index = int(np.argmin(positive_shifted_cmf))
            output += self.alphabet[symbol_index]            
            state_index %= self.n_base_symbols
            state_index *= self.n_base_symbols
            state_index += symbol_index
        self.current_state_index = state_index
            
        return output
    
    def __len__(self) -> int:
        return self.n_symbols
