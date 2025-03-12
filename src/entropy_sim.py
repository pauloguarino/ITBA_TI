from __future__ import annotations

from collections.abc import Iterable
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
        print(f"{self.name} time: {self.stop_time - self.start_time:.3g} s")
    
class Source:
    base_source: Source
    dist: dict[str, float]
    simbols: list[str]
    pmf: list[float]
    cmf: list[float]
    entropy: float
    epsilon: float
    typical_set: set[str]
    
    @property
    def alphabet(self) -> list[str]:
        return self.base_source.simbols

    def __init__(self, dist: dict[str, float], epsilon: float = 0.1, base_source: Source = None):
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
                
    def extend(self, n: int) -> Source:
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
            
        return Source(extended_dist, base_source=self, epsilon=self.epsilon)
            
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


class SourceLite:
    dist: dict[str, float]
    alphabet: list[str]
    pmf: list[float]
    cmf: list[float]
    n_extension: int
    
    def __init__(self, dist: dict[str, float], n_extension: int = 1):
        with Time("Init lite") as _:
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
            
            self.n_extension = n_extension
            
    def probability(self, simbols: str | list[str]) -> float:
        if isinstance(simbols, str):
            probability = 1
            for i in range(self.n_extension):
                probability *= self.dist[simbols[i]]
            return probability
        if isinstance(simbols, Iterable):
            return np.sum([self.probability(simbol) for simbol in simbols])
    
    def entropy(self) -> float:
        with Time("Entropy lite") as _:
            extension_pmf = np.zeros(len(self.alphabet)**self.n_extension)
            for i in range(len(extension_pmf)):
                probability = 1
                var_index = i
                for j in reversed(range(self.n_extension)):
                    simbol_index = var_index // (len(self.alphabet)**j)
                    probability *= self.pmf[simbol_index]
                    var_index -= simbol_index*(len(self.alphabet)**j)
                extension_pmf[i] = probability
            return np.sum(-np.log2(extension_pmf)*extension_pmf)
    
    def __repr__(self) -> str:
        pass
    
    def __call__(self, n: int = 1) -> str:
        pass

def entropy_sim():
    dist = {
        "A": 0.1,
        "B": 0.8,
        "C": 0.05,
        "D": 0.05,
    }
    source = Source(dist)
    source_lite = SourceLite(dist)
    print(source)

    print(f"Entropía de la fuente: {source.entropy:.3g}")
    print(f"Entropía de la fuente lite: {source_lite.entropy():.3g}")
    print(f"Cadena generada: {source(20)}")
    
    # Contrastar 0.9 - 0.1 vs 0.5 - 0.5
    bin_dist = {
        "0": 0.1,
        "1": 0.9,
    }
    n = 22
    epsilon = 0.2 # variar entre 0.2 a 0.4
    bin_source = Source(bin_dist, epsilon=epsilon)
    extended_bin_source = bin_source.extend(n)
    extended_bin_source_lite = SourceLite(bin_dist, n)

    typical_set_size_relation = len(extended_bin_source.typical_set)/len(extended_bin_source)

    print(bin_source)
    print(f"Entropía de la fuente: {bin_source.entropy:.3g}")
    print(f"Entropía de la fuente extendida a {n}: {extended_bin_source.entropy:.3g}")
    print(f"Entropía de la fuente extendida a {n} lite: {extended_bin_source_lite.entropy():.3g}")

    print(f"Tamaño del conjunto típico de epsilon {epsilon}: {len(extended_bin_source.typical_set)}")
    print(f"Relación de tamaño del conjunto típico sobre el total: {typical_set_size_relation*100:.3g} %")
    print(f"Probabilidad acumulada del conjunto típico: {extended_bin_source.probability(extended_bin_source.typical_set):.3g}")

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

    m = 4  # más de 4 necesito mi computadora
    text_source = Source(text_dist)
    extended_text_source = text_source.extend(m)

    print(f"Entropía de la fuente: {text_source.entropy:.3g}")


if __name__ == "__main__":
    entropy_sim()
    