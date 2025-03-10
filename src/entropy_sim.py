from __future__ import annotations

import dit
import numpy as np

class Source(dit.Distribution):
    def __init__(self, outcomes, pmf=None, sample_space=None, base=None, prng=None, sort=True, sparse=True, trim=True, validate=True):
        super().__init__(outcomes, pmf, sample_space, base, prng, sort, sparse, trim, validate)
        
    def extend(self, n: int) -> Source:
        if self._outcome_class is not str:
            raise NotImplementedError("Not implemented for outcome class different from str")
        
        if n < 2:
            return self

        return self._extend(Source(self.outcomes, self.pmf), n)
    
    def _extend(self, extension: Source, n: int) -> Source:
        if n < 2:
            return extension
        
        n -= 1
        extended_pmf = dict()
        for extended_outcome, extended_probability in zip(extension.outcomes, extension.pmf):
            for outcome, probability in zip(self.outcomes, self.pmf):
                extended_pmf[extended_outcome + outcome] = Source._relative_round(extended_probability*probability)
        
        extension = Source(extended_pmf)
        
        return self._extend(extension, n)

    def _relative_round(p: float) -> float:
        rounded_p = p
        
        n = 1
        rounded_p = round(p, n)
        while np.abs(p - rounded_p)/p > 0.0001:
            n += 1
            rounded_p = round(p, n)
        
        return rounded_p

def entropy_sim():
    pmf = {
        "A": 0.1,
        "B": 0.8,
        "C": 0.1,
    }
    source = Source(pmf)

    print(source)
    print(source.rand(20))

    extended_source = source.extend(4)
    print(extended_source)
    print(extended_source.rand(5))

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(source.int_values, source.pmf, 'ro', ms=12, mec='r')
    # ax.vlines(source.int_values, 0, source.pmf, colors='r', lw=4)
    # plt.show()

if __name__ == "__main__":
    entropy_sim()
    