from __future__ import annotations

import dit
import dit.shannon
import numpy as np

class Source_dit(dit.Distribution):
    entropy: float
    base_source: Source_dit
    
    def __init__(self, outcomes, pmf=None, base_source=None, sample_space=None, base=None, prng=None, sort=True, sparse=True, trim=True, validate=False):
        super().__init__(outcomes, pmf, sample_space, base, prng, sort, sparse, trim, validate)
        if self._outcome_class is not str:
            raise NotImplementedError("Not implemented for simbol class different from str")
        
        self.pmf = np.array(self.pmf)
        self.pmf = self.pmf/self.pmf.sum()
        self.entropy = dit.shannon.entropy(self)
        
        self.base_source = self if base_source is None else base_source
        
    def extend(self, n: int) -> Source_dit:
        if n < 2:
            return self

        return self._extend(Source_dit(self.outcomes, self.pmf, base_source=self), n)
    
    def _extend(self, extension: Source_dit, n: int) -> Source_dit:
        if n < 2:
            return extension
        
        n -= 1
        extended_pmf = dict()
        for extended_simbol, extended_probability in zip(extension.outcomes, extension.pmf):
            for simbol, probability in zip(self.outcomes, self.pmf):
                extended_pmf[extended_simbol + simbol] = extended_probability*probability
                if n == 2:
                    extended_pmf[extended_simbol + simbol] = Source_dit._relative_round(extended_probability*probability)
        
        extension = Source_dit(extended_pmf, base_source=self.base_source)
        
        return self._extend(extension, n)

    def _relative_round(p: float) -> float:
        rounded_p = p
        
        n = 1
        rounded_p = round(p, n)
        while np.abs(p - rounded_p)/p > 0.0001:
            n += 1
            rounded_p = round(p, n)
        
        return rounded_p
    
    def get_typical_list(self, epsilon: float) -> list[str]:
        typical_list = []
        for simbol, probability in zip(self.outcomes, self.pmf):
            if 2**(-self.outcome_length()*(self.base_source.entropy + epsilon)) < probability < 2**(-self.outcome_length()*(self.base_source.entropy - epsilon)):
                typical_list.append(simbol)
        
        return typical_list
    
class Source:
    base_source: Source
    pmf_dict: dict[str, float]
    simbols: list[str]
    pmf: list[float]
    entropy: float
    epsilon: float
    typical_set: set[str]
    
    @property
    def alphabet(self) -> list[str]:
        return self.base_source.simbols

    def __init__(self, pmf_dict, epsilon: float = 0.1, base_source: Source = None):
        self.base_source = self if base_source is None else base_source
        self.pmf_dict = pmf_dict
        self.simbols = [simbol for simbol in pmf_dict.keys()]
        self.pmf = np.array([probability for probability in pmf_dict.values()])
        self.pmf = self.pmf/self.pmf.sum()
        for simbol in self.pmf_dict.keys():
            self.pmf_dict[simbol] = self.pmf_dict[simbol]/self.pmf.sum()
        self.entropy = np.sum(-np.log2(self.pmf)*self.pmf)
        
        self.epsilon = epsilon
        self.typical_set = {}
        for simbol, probability in self.pmf_dict.items():
            if 2**(-len(self.alphabet)*(self.base_source.entropy + self.epsilon)) < probability < 2**(-len(self.alphabet)*(self.base_source.entropy - self.epsilon)):
                self.typical_set.add(simbol)

    def extend(self, n: int) -> Source:
        if n < 2:
            return self

        return self._extend(Source(self.pmf_dict, base_source=self), n)
    
    def _extend(self, extension: Source, n: int) -> Source:
        if n < 2:
            return extension
        
        n -= 1
        extended_pmf = dict()
        for extended_simbol, extended_probability in extension.pmf_dict.items():
            for simbol, probability in self.pmf_dict.items():
                extended_pmf[extended_simbol + simbol] = extended_probability*probability
                if n == 2:
                    extended_pmf[extended_simbol + simbol] = Source._relative_round(extended_probability*probability)
        
        extension = Source(extended_pmf, base_source=self.base_source)
        
        return self._extend(extension, n)

    def _relative_round(p: float) -> float:
        rounded_p = p
        
        n = 1
        rounded_p = round(p, n)
        while np.abs(p - rounded_p)/p > 0.0001:
            n += 1
            rounded_p = round(p, n)
        
        return rounded_p

    def __repr__(self) -> str:
        # TODO: dejarlo como la otra
        return f"{(self.pmf_dict)}"


def entropy_sim():
    pmf = {
        "A": 0.1,
        "B": 0.8,
        "C": 0.05,
        "D": 0.05,
    }
    source = Source(pmf)
    print(source)
    print(f"{source.entropy:.4f}")
    # print(source.rand(20))

    extended_source = source.extend(4)
    print(extended_source)
    print(f"{extended_source.entropy:.4f}")
    # print(extended_source.rand(5))
    # epsilon = 0.1
    # print(f"{extended_source.get_typical_list(epsilon)}")
    # print(f"{len(extended_source.get_typical_list(epsilon))/len(extended_source)}")
    # print(f"{(2**(-extended_source.outcome_length()*(source.entropy + epsilon)),
    #       2**(-extended_source.outcome_length()*(source.entropy - epsilon)))}")

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(source.int_values, source.pmf, 'ro', ms=12, mec='r')
    # ax.vlines(source.int_values, 0, source.pmf, colors='r', lw=4)
    # plt.show()

if __name__ == "__main__":
    entropy_sim()
    