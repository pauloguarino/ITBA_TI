from __future__ import annotations

from collections.abc import Iterable
import matplotlib.pyplot as plt
from numba import njit, float32, float64, int32, int64
import numpy as np
import time
from types import TracebackType

class Time:
    name: str
    start_time: float
    stop_time: float
    
    def __init__(self, name: str):
        self.name = name
    
    def __enter__(self) -> Time:
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.stop_time = time.perf_counter()
        # print(f"{self.name} time: {self.stop_time - self.start_time:.3g} s")
    
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
        with Time("Init") as _:
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
        with Time("Extend") as _:
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


@njit(
    [float32(float32[:], int32, int32), float64(float64[:], int32, int32)],
    # parallel=True,
    cache=True,
    )
def fast_entropy(pmf: np.ndarray, alphabet_len: int, n_extension: int) -> float:
    entropy_value = 0
    # extension_pmf = np.zeros(alphabet_len**n_extension)
    for i in range(alphabet_len**n_extension):
        probability = 1
        var_index = i
        for j in range(n_extension - 1, -1, -1):
            simbol_index = var_index // (alphabet_len**j)
            probability *= pmf[simbol_index]
            var_index -= simbol_index*(alphabet_len**j)
        # extension_pmf[i] = probability
        entropy_value += -np.log2(probability)*probability
    # return np.sum(-np.log2(extension_pmf)*extension_pmf)
    return entropy_value

@njit(
    [float32(float32[:], int64[:], int32, int32), float64(float64[:], int64[:], int32, int32)],
    parallel=True,
    cache=True,
    )
def fast_probability(pmf: np.ndarray, simbols: np.ndarray, alphabet_len: int, n_extension: int) -> float:
    probability = np.ones(len(simbols))
    for i in range(len(simbols)):
        var_index = simbols[i]
        for j in range(n_extension - 1, -1, -1):
            simbol_index = var_index // (alphabet_len**j)
            probability[i] *= pmf[simbol_index]
            var_index -= simbol_index*(alphabet_len**j)
    return np.sum(probability)

@njit(
    [int64[:](float32[:], float32, float32, int32, int32), int64[:](float64[:], float32, float32, int32, int32)],
    cache=True,
    )
def fast_typical_set(pmf: np.ndarray, base_entropy: float, epsilon: float, alphabet_len: int, n_extension: int) -> np.ndarray:
    # indexes = np.zeros(alphabet_len**n_extension, int32)
    indexes = []
    for i in range(alphabet_len**n_extension):
        probability = 1
        var_index = i
        for j in range(n_extension - 1, -1, -1):
            simbol_index = var_index // (alphabet_len**j)
            probability *= pmf[simbol_index]
            var_index -= simbol_index*(alphabet_len**j)
        if 2**(-n_extension*(base_entropy + epsilon)) < probability < 2**(-n_extension*(base_entropy - epsilon)):
            # indexes[i] = 1
            indexes.append(i)
    return np.array(indexes, int64)

class Source:
    dist: dict[str, float]
    alphabet: list[str]
    pmf: np.ndarray
    cmf: np.ndarray
    base_entropy: float
    n_extension: int
    null_character = "\0"
    
    def __init__(self, dist: dict[str, float], n_extension: int = 1):
        self.dist = dist
        
        self.alphabet = [simbol for simbol in dist.keys()]
        
        self.pmf = np.array([float(probability) for probability in dist.values()])
        pmf_norm = self.pmf.sum()
        self.pmf /= pmf_norm
        
        self.cmf = np.zeros(len(self.pmf))
        cmf = 0
        for i in range(len(self.pmf)):
            cmf += self.pmf[i]
            self.cmf[i] = cmf
        self.cmf /= cmf
        
        self.base_entropy = np.sum(-np.log2(self.pmf)*self.pmf)
        
        self.n_extension = n_extension
        self.max_simbol_length = max(len(s) for s in self.alphabet)*self.n_extension
                        
    def index_to_simbol(self, index: int) -> str:
        var_index = index
        simbol = ""
        for i in range(self.n_extension - 1, -1, -1):
            simbol_index = var_index // (len(self.alphabet)**i)
            simbol += self.alphabet[simbol_index]
            var_index -= simbol_index*(len(self.alphabet)**i)
        return simbol
    
    def simbol_to_index(self, simbol: str) -> int:
        # funciona solo si el alfabeto de la fuente es un código instantáneo
        padded_simbol = simbol + self.null_character*(self.max_simbol_length - len(simbol))
        index = 0
        string_index = 0
        simbol_index = 0
        while string_index < len(simbol):
            for j in range(len(self.alphabet)):
                alphabet_simbol = self.alphabet[j]
                if padded_simbol[string_index:string_index + len(alphabet_simbol)] == alphabet_simbol:
                    alphabet_simbol_index = j
                    alphabet_simbol_length = len(alphabet_simbol)
            index += alphabet_simbol_index*(len(self.alphabet)**(self.n_extension - simbol_index - 1))
            string_index += alphabet_simbol_length
            simbol_index += 1
        
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
                return fast_probability(self.pmf, np.array(simbols), len(self.alphabet), self.n_extension)
            
    def entropy(self) -> float:
        with Time("Entropy") as _:
            return fast_entropy(self.pmf, len(self.alphabet), self.n_extension)
    
    def typical_set(self, epsilon: float = 0.1) -> set[str]:
        with Time("Typical set") as _:
            typical_set_indexes = fast_typical_set(self.pmf, self.base_entropy, epsilon, len(self.alphabet), self.n_extension)

            return {self.index_to_simbol(index) for index in typical_set_indexes}
    
    def __repr__(self) -> str:
        print_output = "Simbol\t"
        print_output_len = len(print_output) + 5
        while print_output_len < self.max_simbol_length + 1:
            print_output += "\t"
            print_output_len += 6
        print_output += "Probability"
        size = len(self.alphabet)**self.n_extension
        max_print = 10
        if size < max_print:
            for i in range(len(self.alphabet)**self.n_extension):
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
            for i in range(size - max_print//2, size):
                simbol = self.index_to_simbol(i)
                print_output += "\n"
                print_output += simbol
                print_output += "\t"
                print_output += f"{self.probability(simbol):.3g}"
        
        return print_output
    
    def __call__(self, n: int = 1) -> str:
        output = ""
        for i in range(n):
            for j in range(self.n_extension):
                shifted_cmf = self.cmf - np.random.random()
                positive_shifted_cmf = np.where(shifted_cmf < 0, np.inf, shifted_cmf)
                output += self.alphabet[np.argmin(positive_shifted_cmf)]
            
        return output
    
    def __len__(self) -> int:
        return len(self.alphabet)**self.n_extension

def entropy_sim():
    dist = {
        "A": 0.1,
        "B": 0.8,
        "C": 0.05,
        "D": 0.05,
    }
    source = Source(dist)

    print(source)
    print(f"Entropía de la fuente: {source.entropy():.3g}")
    print(f"Cadena generada: {source(20)}")

def typical_set_sim():
    bin_dist = {  # contrastar 0.9 - 0.1 vs 0.5 - 0.5
        "0": 0.1,
        "1": 0.9,
    }
    n = 22  # 22
    epsilon = 0.31 # variar entre 0.2 a 0.4
    bin_source = Source(bin_dist)
    extended_bin_source = Source(bin_dist, n)
    extended_bin_source_typical_set = extended_bin_source.typical_set(epsilon)
    typical_set_size_relation = len(extended_bin_source_typical_set)/len(extended_bin_source)

    # print(extended_bin_source)
    print(f"Entropía de la fuente: {bin_source.entropy():.3g}")
    print(f"Entropía de la fuente extendida a {n}: {extended_bin_source.entropy():.3g}")

    print(f"Tamaño del conjunto típico de epsilon {epsilon}: {len(extended_bin_source_typical_set)}")
    print(f"Relación de tamaño del conjunto típico sobre el total: {typical_set_size_relation*100:.3g} %")
    print(f"Probabilidad acumulada del conjunto típico: {extended_bin_source.probability(extended_bin_source_typical_set):.3g}")

    N = 22
    epsilon = 0.31 # variar entre 0.2 a 0.4, énfasis en 0.3
    
    sources = [Source(bin_dist, i) for i in range(1, N + 1)]
    entropies = [source.entropy() for source in sources]
    typical_sets = [source.typical_set(epsilon) for source in sources]
    typical_sets_lengths = np.array([len(typical_set) for typical_set in typical_sets])
    typical_sets_relative_sizes = np.array([length/len(source) for source, length in zip(sources, typical_sets_lengths)])
    typical_sets_probabilities = np.array([source.probability(typical_set) for source, typical_set in zip(sources, typical_sets)])

    plt.figure()
    plt.plot(range(1, N + 1), entropies)
    plt.show()
    
    plt.figure()
    plt.plot(range(1, N + 1), typical_sets_relative_sizes*100)
    plt.show()
    
    plt.figure()
    plt.plot(range(1, N + 1), typical_sets_probabilities)
    plt.show()

def multicharacter_simbols_sim():
    bin_dist = {  # contrastar 0.9 - 0.1 vs 0.5 - 0.5
        "0": 0.7,
        "10": 0.2,
        "110": 0.08,
        "111": 0.02,
    }
    n = 12  # 22
    epsilon = 0.3 # variar entre 0.2 a 0.4
    bin_source = Source(bin_dist)
    extended_bin_source = Source(bin_dist, n)
    extended_bin_source_typical_set = extended_bin_source.typical_set(epsilon)
    typical_set_size_relation = len(extended_bin_source_typical_set)/len(extended_bin_source)

    # print(extended_bin_source)
    print("Started")
    print(f"Entropía de la fuente: {bin_source.entropy():.3g}")
    print(f"Entropía de la fuente extendida a {n}: {extended_bin_source.entropy():.3g}")

    print(f"Tamaño del conjunto típico de epsilon {epsilon}: {len(extended_bin_source_typical_set)}")
    print(f"Relación de tamaño del conjunto típico sobre el total: {typical_set_size_relation*100:.3g} %")
    print(f"Probabilidad acumulada del conjunto típico: {extended_bin_source.probability(extended_bin_source_typical_set):.3g}")

    N = 12
    epsilon = 0.3 # variar entre 0.2 a 0.4, énfasis en 0.3
    
    sources = [Source(bin_dist, i) for i in range(1, N + 1)]
    entropies = [source.entropy() for source in sources]
    typical_sets = [source.typical_set(epsilon) for source in sources]
    typical_sets_lengths = np.array([len(typical_set) for typical_set in typical_sets])
    typical_sets_relative_sizes = np.array([length/len(source) for source, length in zip(sources, typical_sets_lengths)])
    typical_sets_probabilities = np.array([source.probability(typical_set) for source, typical_set in zip(sources, typical_sets)])

    plt.figure()
    plt.plot(range(1, N + 1), entropies)
    plt.show()
    
    plt.figure()
    plt.plot(range(1, N + 1), typical_sets_relative_sizes*100)
    plt.show()
    
    plt.figure()
    plt.plot(range(1, N + 1), typical_sets_probabilities)
    plt.show()


def text_source_sim():
    text_dist = {
        "A": 1,
        "B": 1,
        "C": 1,
        "D": 1,
        "E": 1,
        "F": 1,
        "G": 1,
        "H": 1,
        "I": 1,
        "J": 1,
        "K": 1,
        "L": 1,
        "M": 1,
        "N": 1,
        "Ñ": 1,
        "O": 1,
        "P": 1,
        "Q": 1,
        "R": 1,
        "S": 1,
        "T": 1,
        "U": 1,
        "V": 1,
        "W": 1,
        "X": 1,
        "Y": 1,
        "Z": 1,
        ".": 1,
        ",": 1,
        ":": 1,
        "-": 1,
        " ": 1,
    }

    m = 5  # más de 6 tarda muchos minutos en calcular la entropía
    text_source = Source(text_dist)
    extended_text_source = Source(text_dist, m)

    print(f"Entropía de la fuente: {text_source.entropy():.3g}")
    print(f"Entropía de la fuente extendida: {extended_text_source.entropy():.3g}")
    print(extended_text_source(10))

if __name__ == "__main__":
    # entropy_sim()
    # typical_set_sim()
    multicharacter_simbols_sim()
    # text_source_sim()
    