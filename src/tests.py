from ti_modules import information
import numpy as np

def tests():
    alphabet = ["0", "1"]
    memory_pmf = np.array([[0.75, 0.25], [0.95, 0.05], [0, 1], [0.5, 0.5]])
    estimated_pmf = np.array([[0, 1], [0.93, 0.0705], [0.0227, 0.977], [0.564, 0.436]])
    memory_source = information.MemorySource((alphabet, memory_pmf))
    extended_memory_source = information.MemorySource((alphabet, memory_pmf), 2)
    extended_estimated_source = information.MemorySource((alphabet, estimated_pmf), 2)
    print(memory_source)
    print(extended_memory_source)
    print(extended_memory_source.unconditional_source())
    print(extended_estimated_source)
    print(extended_estimated_source.unconditional_source())
    print(np.array([extended_memory_source.probability(i) - extended_estimated_source.probability(i) for i in range(4)]))


if __name__ == "__main__":
    tests()
    