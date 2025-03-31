import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from ti_modules import information

def entropy_sim():
    dist = {
        "A": 0.1,
        "B": 0.8,
        "C": 0.05,
        "D": 0.05,
    }
    n_generated = 20
    
    source = information.Source(dist)

    print(source)
    print(f"Entropía de la fuente: {source.entropy():.3g} bits")
    print(f"Cadena generada: {source(n_generated)}")

def typical_set_sim():
    bin_dist = {  # contrastar 0.9 - 0.1 vs 0.5 - 0.5
        "0": 0.1,
        "1": 0.9,
    }
    n_extension = 22  # 22
    epsilon = 0.31 # variar entre 0.2 a 0.4
    bin_source = information.Source(bin_dist)
    extended_bin_source = information.Source(bin_dist, n_extension)
    extended_bin_source_typical_set = extended_bin_source.typical_set(epsilon)
    typical_set_size_relation = len(extended_bin_source_typical_set)/len(extended_bin_source)

    print(extended_bin_source)
    
    print(f"Entropía de la fuente: {bin_source.entropy():.3g} bits")
    print(f"Entropía de la fuente extendida a {n_extension}: {extended_bin_source.entropy():.3g} bits")

    print(f"Tamaño del conjunto típico de epsilon {epsilon}: {len(extended_bin_source_typical_set)}")
    print(f"Relación de tamaño del conjunto típico sobre el total: {typical_set_size_relation*100:.3g} %")
    print(f"Probabilidad acumulada del conjunto típico: {extended_bin_source.probability(extended_bin_source_typical_set):.3g}")

    n_extension_max = 22
    epsilon = 0.31 # variar entre 0.2 a 0.4, énfasis en 0.3
    
    sources = [information.Source(bin_dist, i) for i in range(1, n_extension_max + 1)]
    entropies = [source.entropy() for source in sources]
    typical_sets = [source.typical_set(epsilon) for source in tqdm(sources)]
    typical_sets_lengths = np.array([len(typical_set) for typical_set in typical_sets])
    typical_sets_relative_sizes = np.array([length/len(source) for source, length in zip(sources, typical_sets_lengths)])
    typical_sets_probabilities = np.array([source.probability(typical_set) for source, typical_set in zip(sources, typical_sets)])

    theoretical_typical_sets_relative_sizes = np.array([2**(source.n_extension*source.base_entropy)/len(source) for source in sources])
    theoretical_typical_sets_probabilities_limit = 1 - epsilon

    plt.figure()
    plt.plot(range(1, n_extension_max + 1), entropies)
    plt.show()
    
    plt.figure()
    plt.plot(range(1, n_extension_max + 1), typical_sets_relative_sizes*100)
    plt.plot(range(1, n_extension_max + 1), theoretical_typical_sets_relative_sizes*100)
    plt.show()
    
    plt.figure()
    plt.plot(range(1, n_extension_max + 1), typical_sets_probabilities)
    plt.plot(range(1, n_extension_max + 1), [theoretical_typical_sets_probabilities_limit for i in range(n_extension_max)])
    plt.show()

def multicharacter_simbols_sim():
    bin_dist = {
        "000": 0.4,
        "100": 0.3,
        "110": 0.2,
        "111": 0.1,
    }
    n_extension = 12  # 22
    epsilon = 0.3 # variar entre 0.2 a 0.4
    bin_source = information.Source(bin_dist)
    extended_bin_source = information.Source(bin_dist, n_extension)
    extended_bin_source_typical_set = extended_bin_source.typical_set(epsilon)
    typical_set_size_relation = len(extended_bin_source_typical_set)/len(extended_bin_source)

    # print(extended_bin_source)
    print(f"Entropía de la fuente: {bin_source.entropy():.3g} bits")
    print(f"Entropía de la fuente extendida a {n_extension}: {extended_bin_source.entropy():.3g} bits")

    print(f"Tamaño del conjunto típico de epsilon {epsilon}: {len(extended_bin_source_typical_set)}")
    print(f"Relación de tamaño del conjunto típico sobre el total: {typical_set_size_relation*100:.3g} %")
    print(f"Probabilidad acumulada del conjunto típico: {extended_bin_source.probability(extended_bin_source_typical_set):.3g}")

    n_extension_max = 12
    epsilon = 0.3 # variar entre 0.2 a 0.4, énfasis en 0.3
    
    sources = [information.Source(bin_dist, i) for i in range(1, n_extension_max + 1)]
    entropies = [source.entropy() for source in sources]
    typical_sets = [source.typical_set(epsilon) for source in tqdm(sources)]
    typical_sets_lengths = np.array([len(typical_set) for typical_set in typical_sets])
    typical_sets_relative_sizes = np.array([length/len(source) for source, length in zip(sources, typical_sets_lengths)])
    typical_sets_probabilities = np.array([source.probability(typical_set) for source, typical_set in zip(sources, typical_sets)])

    theoretical_typical_sets_relative_sizes = np.array([2**(source.n_extension*source.base_entropy)/len(source) for source in sources])
    theoretical_typical_sets_probabilities_limit = 1 - epsilon

    plt.figure()
    plt.plot(range(1, n_extension_max + 1), entropies)
    plt.show()
    
    plt.figure()
    plt.plot(range(1, n_extension_max + 1), typical_sets_relative_sizes*100)
    plt.plot(range(1, n_extension_max + 1), theoretical_typical_sets_relative_sizes*100)
    plt.show()
    
    plt.figure()
    plt.plot(range(1, n_extension_max + 1), typical_sets_probabilities)
    plt.plot(range(1, n_extension_max + 1), [theoretical_typical_sets_probabilities_limit for i in range(n_extension_max)])
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
    n_extension = 5  # más de 6 tarda muchos minutos en calcular la entropía
    n_generated = 10
    text_source = information.Source(text_dist)
    extended_text_source = information.Source(text_dist, n_extension)

    print(f"Entropía de la fuente: {text_source.entropy():.3g} bits")
    print(f"Entropía de la fuente extendida: {extended_text_source.entropy():.3g} bits")
    print(extended_text_source(n_generated))

def memory_source_sim():
    alphabet = ["0", "1"]
    # pmf = np.array([[0.1, 0.9], [0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])
    pmf = np.array([[0.75, 0.25], [0.95, 0.05], [0, 1], [0.5, 0.5]])
    # pmf = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.45, 0.55], [0.85, 0.15]])
    n_extension = 8
    n_generated = 50
    source = information.MemorySource((alphabet, pmf), n_extension)
    unconditional_source = source.unconditional_source()

    print(f"Fuente con memoria de orden {source.memory} de extensión {source.n_extension}")
    print(source)
    
    print(f"Matriz de transición: {np.array2string(source.transition_matrix, precision=3, floatmode='fixed')}")
    print(f"Probabilidad de estados: {np.array2string(source.state_pmf, precision=3, floatmode='fixed')}")
    print(f"Límite de la matriz de transición: {np.array2string(np.linalg.matrix_power(source.transition_matrix, 100), precision=3, floatmode='fixed')}")
    
    print("Fuente afín")
    print(unconditional_source)
    print(f"Probabilidad incondicional de la fuente no extendida: {np.array2string(source.unconditional_simple_pmf, precision=3, floatmode='fixed')}")

    print(f"Entropía de la fuente con memoria de orden {source.memory} de extensión {source.n_extension}: {source.entropy():.3g} bits")
    print(f"Entropía de la fuente afín: {unconditional_source.entropy():.3g} bits")

    print(f"Cadena generada: {source(n_generated)}")
    print(f"Cadena generada con la fuente afín: {unconditional_source(n_generated)}")

def memoryless_source_sim():
    alphabet = ["0", "1"]
    pmf = np.array([[0.1, 0.9], [0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])
    # pmf = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.45, 0.55], [0.85, 0.15]])
    n_extension_max = 15
    
    memory_sources = [information.MemorySource((alphabet, pmf), i) for i in range(1, n_extension_max + 1)]
    memoryless_sources = [memory_source.unconditional_source() for memory_source in memory_sources]
    memory_entropies = np.array([source.entropy() for source in memory_sources])
    memoryless_entropies = np.array([source.entropy() for source in memoryless_sources])

    plt.figure()
    plt.plot(range(1, n_extension_max + 1), memoryless_entropies - memory_entropies)
    plt.show()

def compression_sim():
    alphabet = ["0", "1"]
    pmf = np.array([0.1, 0.9])
    # memory_pmf = np.array([[0.1, 0.9], [0.3, 0.7], [0.8, 0.2], [0.99, 0.01]])
    memory_pmf = np.array([[0.75, 0.25], [0.95, 0.05], [0, 1], [0.5, 0.5]])
    output_size = 8000*2**10
    fixed_bits_per_byte = 3
    
    source = information.Source((alphabet, pmf), 8)
    memory_source = information.MemorySource((alphabet, memory_pmf), 8)
    memoryless_source = memory_source.unconditional_source()
    source_entropy_per_simbol = source.entropy_per_simbol
    memory_source_entropy_per_simbol = memory_source.entropy_per_simbol
    memoryless_source_entropy_per_simbol = memoryless_source.entropy_per_simbol
    
    print(f"Entropía por símbolo de la fuente: {source_entropy_per_simbol:.3g} bits")
    print(f"Entropía por símbolo de la fuente con memoria: {memory_source_entropy_per_simbol:.3g} bits")
    print(f"Entropía por símbolo de la fuente afín: {memoryless_source_entropy_per_simbol:.3g} bits")
    
    current_dir_path = Path.cwd()
    output_dir_path = current_dir_path.joinpath("outputs")
    source_output_path = output_dir_path.joinpath("entropy_compression_sim_source_output")
    memory_source_output_path = output_dir_path.joinpath("entropy_compression_sim_memory_source_output")
    memoryless_source_output_path = output_dir_path.joinpath("entropy_compression_sim_memoryless_source_output")
    random_output_path = output_dir_path.joinpath("entropy_compression_sim_random_output")
    semirandom_output_path = output_dir_path.joinpath("entropy_compression_sim_semirandom_output")
    
    expected_source_output_compressed_size = output_size*source_entropy_per_simbol
    expected_memory_source_output_compressed_size = output_size*memory_source_entropy_per_simbol
    expected_memoryless_source_output_compressed_size = output_size*memoryless_source_entropy_per_simbol
    mask = 2**(8 - fixed_bits_per_byte) - 1
    expected_semirandom_output_compressed_size = output_size*(8 - fixed_bits_per_byte)/8

    print(f"Tamaño de los archivos de salida: {output_size/1024:.0f} Kb")
    print(f"Tamaño esperado del archivo generado por la fuente comprimido: {expected_source_output_compressed_size/1024:.0f} Kb")
    print(f"Tamaño esperado del archivo generado por la fuente con memoria comprimido: {expected_memory_source_output_compressed_size/1024:.0f} Kb")
    print(f"Tamaño esperado del archivo generado por la fuente afín comprimido: {expected_memoryless_source_output_compressed_size/1024:.0f} Kb")
    print(f"Tamaño esperado del archivo semialeatorio comprimido: {expected_semirandom_output_compressed_size/1024:.0f} Kb")
    
    # with open(source_output_path, "bw") as file:
    #     for _ in tqdm(range(output_size)):
    #         file.write(bytes([source.simbol_to_index(source())]))

    # me da demasiado pesado este, puede ser porque codifica de a bytes, tendría que tener memoria de más de 1 byte (múltiples bytes)
    with open(memory_source_output_path, "bw") as file:
        for _ in tqdm(range(output_size)):
            file.write(bytes([memory_source.simbol_to_index(memory_source())]))

    with open(memoryless_source_output_path, "bw") as file:
        for _ in tqdm(range(output_size)):
            file.write(bytes([memoryless_source.simbol_to_index(memoryless_source())]))       
    
    # with open(random_output_path, "bw") as random_file:
    #     with open(semirandom_output_path, "bw") as semirandom_file:
    #         for _ in tqdm(range(output_size)):
    #             byte = np.random.randint(256)
    #             random_file.write(bytes([byte]))
    #             semirandom_file.write(bytes([byte & mask]))

def memory_typical_set_sim():
    alphabet = ["0", "1"]
    # pmf = np.array([[0.1, 0.9], [0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])
    pmf = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.45, 0.55], [0.85, 0.15]])
    n_extension_max = 4
    
    n_extension = 15
    epsilon = 0.3 # variar entre 0.2 a 0.4
    bin_source = information.MemorySource((alphabet, pmf))
    extended_bin_source = information.MemorySource((alphabet, pmf), n_extension)
    extended_bin_source_typical_set = extended_bin_source.typical_set(epsilon)
    typical_set_size_relation = len(extended_bin_source_typical_set)/len(extended_bin_source)

    print(extended_bin_source)
    
    print(f"Entropía de la fuente: {bin_source.entropy():.3g} bits")
    print(f"Entropía de la fuente extendida a {n_extension}: {extended_bin_source.entropy():.3g} bits")

    print(f"Tamaño del conjunto típico de epsilon {epsilon}: {len(extended_bin_source_typical_set)}")
    print(f"Relación de tamaño del conjunto típico sobre el total: {typical_set_size_relation*100:.3g} %")
    print(f"Probabilidad acumulada del conjunto típico: {extended_bin_source.probability(extended_bin_source_typical_set):.3g}")

    n_extension_max = 20
    epsilon = 0.2 # variar entre 0.2 a 0.4, énfasis en 0.3
    
    sources = [information.MemorySource((alphabet, pmf), i) for i in range(1, n_extension_max + 1)]
    entropies = [source.entropy() for source in sources]
    typical_sets = [source.typical_set(epsilon) for source in tqdm(sources)]
    typical_sets_lengths = np.array([len(typical_set) for typical_set in typical_sets])
    typical_sets_relative_sizes = np.array([length/len(source) for source, length in zip(sources, typical_sets_lengths)])
    typical_sets_probabilities = np.array([source.probability(typical_set) for source, typical_set in zip(sources, typical_sets)])

    theoretical_typical_sets_relative_sizes = np.array([2**(source.n_extension*source.base_entropy)/len(source) for source in sources])
    theoretical_typical_sets_probabilities_limit = 1 - epsilon

    plt.figure()
    plt.plot(range(1, n_extension_max + 1), entropies)
    plt.show()
    
    plt.figure()
    plt.plot(range(1, n_extension_max + 1), typical_sets_relative_sizes*100)
    plt.plot(range(1, n_extension_max + 1), theoretical_typical_sets_relative_sizes*100)
    plt.show()
    
    plt.figure()
    plt.plot(range(1, n_extension_max + 1), typical_sets_probabilities)
    plt.plot(range(1, n_extension_max + 1), [theoretical_typical_sets_probabilities_limit for i in range(n_extension_max)])
    plt.show()

if __name__ == "__main__":
    # entropy_sim()
    # typical_set_sim()
    # multicharacter_simbols_sim()
    # text_source_sim()
    # memory_source_sim()
    # memoryless_source_sim()
    # compression_sim()
    memory_typical_set_sim()
    pass
    