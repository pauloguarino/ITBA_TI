import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ti_modules import information

# Simulación de una fuente de 4 símbolos con una distribución arbitraria
def entropy_sim():
    dist = {  # la distribución se puede definir con un diccionario
        "A": 0.1,
        "B": 0.8,
        "C": 0.05,
        "D": 0.05,
    }
    n_generated = 20
    
    source = information.Source(dist)  # define la fuente en base a la distribución

    print(source)  # imprime los símbolos de la fuente junto con su probabilidad
    print(f"Entropía de la fuente: {source.entropy():.3g} bits")  # calcula la entropía de la fuente en bits por defecto
    print(f"Cadena generada: {source(n_generated)}")  # genera n_generated símbolos aleatoriamente en base a la distribución

# Simulación del conjunto típico de una fuente y de sus propiedades en función del número de extensión de la fuente
def typical_set_sim():
    bin_dist = {  # contrastar 0.9 - 0.1 vs 0.5 - 0.5
        "0": 0.1,
        "1": 0.9,
    }
    n_extension = 22  # no subir más que este valor (o ir subiendo de a poco) por cuestiones de recursos, particularmente de tiempo de cómputo
    epsilon = 0.31  # variar entre 0.2 a 0.4
    
    bin_source = information.Source(bin_dist)  # define la fuente binaria
    extended_bin_source = information.Source(bin_dist, n_extension)  # define la extensión de la fuente de largo n_extension
    
    extended_bin_source_typical_set = extended_bin_source.typical_set(epsilon)  # calcula el conjunto típico
    typical_set_size_relation = len(extended_bin_source_typical_set)/len(extended_bin_source)  # calcula la relación entre el tamaño del conjunto típico y el total de símbolos

    print(extended_bin_source)
    
    print(f"Entropía de la fuente: {bin_source.entropy():.3g} bits")
    print(f"Entropía de la fuente extendida a {n_extension}: {extended_bin_source.entropy():.3g} bits")

    print(f"Tamaño del conjunto típico de epsilon {epsilon}: {len(extended_bin_source_typical_set)}")
    print(f"Relación de tamaño del conjunto típico sobre el total: {typical_set_size_relation*100:.3g} %")
    print(f"Probabilidad acumulada del conjunto típico: {extended_bin_source.probability(extended_bin_source_typical_set):.3g}")  # calcula la probabilidad conjunta del conjunto típico

    n_extension_max = 22  # simula lo mismo que antes pero para todos los números de extensión hasta n_extension_max
    epsilon = 0.31  # variar entre 0.2 a 0.4, énfasis en 0.3
    
    sources = [information.Source(bin_dist, i) for i in range(1, n_extension_max + 1)]
    entropies = [source.entropy() for source in sources]
    typical_sets = [source.typical_set(epsilon) for source in tqdm(sources)]
    typical_sets_lengths = np.array([len(typical_set) for typical_set in typical_sets])
    typical_sets_relative_sizes = np.array([length/len(source) for source, length in zip(sources, typical_sets_lengths)])
    typical_sets_probabilities = np.array([source.probability(typical_set) for source, typical_set in zip(sources, typical_sets)])

    theoretical_typical_sets_relative_sizes = np.array([2**(source.n_extension*source.base_entropy)/len(source) for source in sources])
    theoretical_typical_sets_probabilities_limit = 1 - epsilon

    plt.figure()
    plt.plot(range(1, n_extension_max + 1), entropies)  # grafica la entropía en función del número de extensión
    plt.xlabel("Número de extensión")
    plt.ylabel("Entropía [bits]")
    plt.show()
    
    plt.figure()
    plt.plot(range(1, n_extension_max + 1), typical_sets_relative_sizes*100)  # grafica el tamaño relativo del conjunto típico en función del número de extensión
    plt.plot(range(1, n_extension_max + 1), theoretical_typical_sets_relative_sizes*100)  # grafica el valor teórico del mismo que se debería parecer a medida que crece el número de extensión
    plt.show()
    
    plt.figure()
    plt.plot(range(1, n_extension_max + 1), typical_sets_probabilities)  # grafica la probabilidad conjunta del conjunto típico en función del número de extensión
    plt.plot(range(1, n_extension_max + 1), [theoretical_typical_sets_probabilities_limit for i in range(n_extension_max)])  # grafica el límite inferior teórico del mismo que se debería superar con un número de extensión suficientemente grande
    plt.show()

# Simulación similar a la primera pero con una fuente con símbolos de caracteres múltiples
def multicharacter_symbols_sim():
    bin_dist = {  # se pueden definir símbolos de múltiples caracteres, pero no es recomendado (ni particularmente útil), y todos los símbolos deben tener el mismo largo por cuestiones de implementación
        "000": 0.4,
        "100": 0.3,
        "110": 0.2,
        "111": 0.1,
    }
    n_generated = 7
    n_extension = 12
    
    multicharacter_source = information.Source(bin_dist)
    
    extended_multicharacter_source = information.Source(bin_dist, n_extension)

    print(f"Entropía de la fuente: {multicharacter_source.entropy():.3g} bits")
    print(f"Entropía de la fuente extendida a {n_extension}: {extended_multicharacter_source.entropy():.3g} bits")

    print(multicharacter_source)
    print(f"Entropía de la fuente: {multicharacter_source.entropy():.3g} bits")
    print(f"Cadena generada: {multicharacter_source(n_generated)}")

# Simulación de una fuente con 32 caracteres de escritura
def text_source_sim():
    text_dist = {  # fuente equiprobable (no es representativa del lenguaje), las probabilidades son normalizadas en la creación del objeto
        "A": 1.0,
        "B": 1.0,
        "C": 1.0,
        "D": 1.0,
        "E": 1.0,
        "F": 1.0,
        "G": 1.0,
        "H": 1.0,
        "I": 1.0,
        "J": 1.0,
        "K": 1.0,
        "L": 1.0,
        "M": 1.0,
        "N": 1.0,
        "Ñ": 1.0,
        "O": 1.0,
        "P": 1.0,
        "Q": 1.0,
        "R": 1.0,
        "S": 1.0,
        "T": 1.0,
        "U": 1.0,
        "V": 1.0,
        "W": 1.0,
        "X": 1.0,
        "Y": 1.0,
        "Z": 1.0,
        ".": 1.0,
        ",": 1.0,
        ":": 1.0,
        "-": 1.0,
        " ": 1.0,
    }
    n_extension = 5  # más de 6 tarda muchos minutos en calcular la entropía
    n_generated = 10
    text_source = information.Source(text_dist)
    extended_text_source = information.Source(text_dist, n_extension)

    print(f"Entropía de la fuente: {text_source.entropy():.3g} bits")
    print(f"Entropía de la fuente extendida: {extended_text_source.entropy():.3g} bits")
    print(extended_text_source(n_generated))

# Simulación de una fuente con memoria
def memory_source_sim():
    alphabet = ["0", "1"]  # para las fuentes con memoria, se define el alfabeto y la distribución por separado
    pmf = np.array([[0.1, 0.9], [0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])  # el arreglo tiene tantos elementos como estados (el número de estados tiene que ser potencia de la cantidad de símbolos en el alfabeto), y cada elemento tiene la distribución de probabilidad de cada símbolo condicionado a ese estado
    # pmf = np.array([[0.75, 0.25], [0.95, 0.05], [0, 1], [0.5, 0.5]])  # distribución alternativa para probar de memoria 2
    # pmf = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.45, 0.55], [0.85, 0.15]])  # distribución alternativa para probar de memoria 3
    
    n_extension = 8
    n_generated = 50
    
    memory_source = information.MemorySource((alphabet, pmf))  # fuente con memoria
    extended_memory_source = information.MemorySource(memory_source, n_extension)  # extensión de la fuente con memoria
    unconditional_extended_source = extended_memory_source.unconditional_source()  # fuente afín de la extensión de la fuente con memoria
    extended_unconditional_source = information.Source(memory_source.unconditional_source(), n_extension)  # extensión de la fuente afín de la fuente con memoria

    print(f"Fuente con memoria de orden {memory_source.memory}")
    print(memory_source)

    print(f"Extensión de orden {extended_memory_source.n_extension} de la fuente con memoria de orden {extended_memory_source.memory}")
    print(extended_memory_source)
    
    print(f"Matriz de transición:\n{np.array2string(extended_memory_source.transition_matrix, precision=3, floatmode='fixed')}")
    print(f"Probabilidad de estados: {np.array2string(extended_memory_source.state_pmf, precision=3, floatmode='fixed')}")
    print(f"Límite de la matriz de transición:\n{np.array2string(np.linalg.matrix_power(extended_memory_source.transition_matrix, 100), precision=3, floatmode='fixed')}")  # cada fila del límite de la matriz de transición es igual al vector de probabilidad de estados si 
    
    print("Fuente afín de la extensión de la fuente con memoria")
    print(unconditional_extended_source)
    
    print(f"Probabilidad incondicional de la fuente con memoria: {np.array2string(memory_source.unconditional_source().pmf, precision=3, floatmode='fixed')}")
    print("Extensión de la fuente afín de la fuente con memoria no extendida")
    print(extended_unconditional_source)

    print(f"Entropía de la fuente con memoria: {memory_source.entropy():.3g} bits")
    print(f"Entropía de la extensión de la fuente con memoria: {extended_memory_source.entropy():.3g} bits")
    print(f"Entropía de la fuente afín de la extensión de la fuente con memoria: {unconditional_extended_source.entropy():.3g} bits")
    print(f"Entropía de la extensión de la fuente afín de la fuente con memoria: {extended_unconditional_source.entropy():.3g} bits")

    print(f"Cadena generada con la extensión de la fuente con memoria: {extended_memory_source(n_generated)}")
    print(f"Cadena generada con la fuente afín de la extensión de la fuente con memoria: {unconditional_extended_source(n_generated)}")

# Simulación de la relación (la diferencia) entre las entropías de una fuente con memoria y su fuente afín
def memoryless_source_sim():
    alphabet = ["0", "1"]
    memory = 4  # no subir más que este valor (o ir subiendo de a poco) por cuestiones de recursos, particularmente de tiempo de cómputo
    pmf = np.random.uniform(0, 1, (2**memory, 2))  # distribución generada aleatoriamente
    n_extension_max = 15
    
    memory_sources = [information.MemorySource((alphabet, pmf), i) for i in range(1, n_extension_max + 1)]
    memoryless_sources = [memory_source.unconditional_source() for memory_source in memory_sources]
    memory_entropies = np.array([source.entropy() for source in memory_sources])
    memoryless_entropies = np.array([source.entropy() for source in memoryless_sources])

    plt.figure()
    plt.plot(range(1, n_extension_max + 1), memoryless_entropies - memory_entropies)  # grafica la diferencia entre la entropía de la fuente con memoria y la de su fuente afín en función del número de extensión
    plt.show()

# Simulación del conjunto típico de una fuente con memoria y de sus propiedades en función del número de extensión de la fuente
def memory_typical_set_sim():
    alphabet = ["0", "1"]
    # pmf = np.array([[0.1, 0.9], [0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])  # distribución alternativa para probar
    pmf = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.45, 0.55], [0.85, 0.15]])
    
    n_extension = 14
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

# Simulación de los límites de compresión de 5 archivos binarios, 3 generados por 3 fuentes distintas, uno aleatorio, y uno semialeatorio (3 bits por byte fijos, y 5 bits por byte generados con distribución uniforme)
def compression_sim():
    alphabet = ["0", "1"]
    pmf = np.array([0.1, 0.9])
    memory_pmf = np.array([[0.1, 0.9], [0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
    # memory_pmf = np.array([[0.75, 0.25], [0.95, 0.05], [0, 1], [0.5, 0.5]])  # distribución alternativa para probar
    
    output_bytes = 8000*2**10  # 8000 kB
    fixed_bits_per_byte = 3
    
    source = information.Source((alphabet, pmf), 8)
    memory_source = information.MemorySource((alphabet, memory_pmf), 8)
    memoryless_source = memory_source.unconditional_source()
    
    source_entropy_per_symbol = source.entropy_per_symbol
    memory_source_entropy_per_symbol = memory_source.entropy_per_symbol
    memoryless_source_entropy_per_symbol = memoryless_source.entropy_per_symbol
    
    print(memory_source)
    
    print(f"Entropía por símbolo de la fuente: {source_entropy_per_symbol:.3g} bits")
    print(f"Entropía por símbolo de la fuente con memoria: {memory_source_entropy_per_symbol:.3g} bits")
    print(f"Entropía por símbolo de la fuente afín: {memoryless_source_entropy_per_symbol:.3g} bits")
    
    current_dir_path = Path.cwd()
    output_dir_path = current_dir_path.joinpath("output")
    if not output_dir_path.exists():
        os.mkdir(output_dir_path)  # crea la carpeta de output si no existe

    source_output_path = output_dir_path.joinpath("entropy_compression_sim_source_output")
    memory_source_output_path = output_dir_path.joinpath("entropy_compression_sim_memory_source_output")
    memoryless_source_output_path = output_dir_path.joinpath("entropy_compression_sim_memoryless_source_output")
    random_output_path = output_dir_path.joinpath("entropy_compression_sim_random_output")
    semirandom_output_path = output_dir_path.joinpath("entropy_compression_sim_semirandom_output")
    
    expected_source_output_compressed_size = output_bytes*source_entropy_per_symbol
    expected_memory_source_output_compressed_size = output_bytes*memory_source_entropy_per_symbol
    expected_memoryless_source_output_compressed_size = output_bytes*memoryless_source_entropy_per_symbol
    mask = 2**(8 - fixed_bits_per_byte) - 1
    expected_semirandom_output_compressed_size = output_bytes*(8 - fixed_bits_per_byte)/8

    print(f"Tamaño de los archivos de salida: {output_bytes/1024:.0f} Kb")
    print(f"Tamaño esperado del archivo generado por la fuente comprimido: {expected_source_output_compressed_size/1024:.0f} Kb")
    print(f"Tamaño esperado del archivo generado por la fuente con memoria comprimido: {expected_memory_source_output_compressed_size/1024:.0f} Kb")
    print(f"Tamaño esperado del archivo generado por la fuente afín comprimido: {expected_memoryless_source_output_compressed_size/1024:.0f} Kb")
    print(f"Tamaño esperado del archivo semialeatorio comprimido: {expected_semirandom_output_compressed_size/1024:.0f} Kb")
    
    with open(source_output_path, "bw") as file:
        for _ in tqdm(range(output_bytes)):
            file.write(bytes([source.symbol_to_index(source())]))

    with open(memory_source_output_path, "bw") as file:
        for _ in tqdm(range(output_bytes)):
            file.write(bytes([memory_source.symbol_to_index(memory_source())]))

    with open(memoryless_source_output_path, "bw") as file:
        for _ in tqdm(range(output_bytes)):
            file.write(bytes([memoryless_source.symbol_to_index(memoryless_source())]))  
    
    with open(random_output_path, "bw") as random_file:
        with open(semirandom_output_path, "bw") as semirandom_file:
            for _ in tqdm(range(output_bytes)):
                byte = np.random.randint(256)
                random_file.write(bytes([byte]))
                semirandom_file.write(bytes([byte & mask]))

# Cálculo de la entroopía estadística de los archivos generados en la simulación de compresión (las distribuciones tienen que ser las mismas para que coincidan los resultados)
def entropy_rate_sim():
    alphabet = ["0", "1"]
    pmf = np.array([0.1, 0.9])
    memory_pmf = np.array([[0.1, 0.9], [0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
    # memory_pmf = np.array([[0.75, 0.25], [0.95, 0.05], [0, 1], [0.5, 0.5]])  # distribución alternativa para probar
    output_size = 8000*2**10
    
    source = information.Source((alphabet, pmf))
    memory_source = information.MemorySource((alphabet, memory_pmf))
    memoryless_source = memory_source.unconditional_source()

    current_dir_path = Path.cwd()
    input_dir_path = current_dir_path.joinpath("output")
    source_input_path = input_dir_path.joinpath("entropy_compression_sim_source_output")
    memory_source_input_path = input_dir_path.joinpath("entropy_compression_sim_memory_source_output")
    memoryless_source_input_path = input_dir_path.joinpath("entropy_compression_sim_memoryless_source_output")

    estimated_pmf = np.zeros(pmf.shape)
    n_bits = 0
    with open(source_input_path, "br") as source_input_file:
        for _ in tqdm(range(output_size)):
            byte = source_input_file.read(1)
            byte_string = f"{int.from_bytes(byte):08b}"
            for bit in byte_string:
                symbol_index = 0 if bit == "0" else 1
                estimated_pmf[symbol_index] += 1
                n_bits += 1
                
        estimated_pmf /= n_bits
        estimated_source = information.Source((alphabet, estimated_pmf))
        estimated_entropy = estimated_source.entropy()
        print("Fuente sin memoria")
        print(source)
        print(f"Entropía de la fuente: {source.entropy():.3g} bits")
        print("Estimación")
        print(estimated_source)
        print(f"Entropía estimada de la fuente: {estimated_entropy:.3g}")

    estimated_pmf = np.zeros(memory_pmf.shape)
    n_bits = 0
    current_state = ""
    with open(memory_source_input_path, "br") as memory_source_input_file:
        for _ in tqdm(range(output_size)):
            byte = memory_source_input_file.read(1)
            byte_string = f"{int.from_bytes(byte):08b}"
            for bit in byte_string:
                if len(current_state) < memory_source.memory:
                    current_state += bit
                else:
                    state_index = memory_source.state_to_index(current_state)
                    symbol_index = 0 if bit == "0" else 1
                    estimated_pmf[state_index, symbol_index] += 1
                    n_bits += 1
                    current_state += bit
                    current_state = current_state[-memory_source.memory:]
                
        estimated_pmf /= n_bits
        estimated_source = information.MemorySource((alphabet, estimated_pmf))
        estimated_entropy = estimated_source.entropy()
        print("Fuente con memoria")
        print(memory_source)
        print(f"Entropía de la fuente con memoria: {memory_source.entropy():.3g} bits")
        print("Fuente estimada")
        print(estimated_source)
        print(f"Entropía estimada de la fuente con memoria: {estimated_entropy:.3g}")

    estimated_pmf = np.zeros(pmf.shape)
    n_bits = 0
    with open(memory_source_input_path, "br") as memory_source_input_file:
        for _ in tqdm(range(output_size)):
            byte = memory_source_input_file.read(1)
            byte_string = f"{int.from_bytes(byte):08b}"
            for bit in byte_string:
                symbol_index = 0 if bit == "0" else 1
                estimated_pmf[symbol_index] += 1
                n_bits += 1
                
        estimated_pmf /= n_bits
        estimated_source = information.Source((alphabet, estimated_pmf))
        estimated_entropy = estimated_source.entropy()
        print("Fuente con memoria estimada como si no tuviese memoria")
        print(estimated_source)
        print(f"Entropía estimada de la fuente: {estimated_entropy:.3g}")

    estimated_pmf = np.zeros(pmf.shape)
    n_bits = 0
    with open(memoryless_source_input_path, "br") as memoryless_source_input_file:
        for _ in tqdm(range(output_size)):
            byte = memoryless_source_input_file.read(1)
            byte_string = f"{int.from_bytes(byte):08b}"
            for bit in byte_string:
                symbol_index = 0 if bit == "0" else 1
                estimated_pmf[symbol_index] += 1
                n_bits += 1
                
        estimated_pmf /= n_bits
        estimated_source = information.Source((alphabet, estimated_pmf))
        estimated_entropy = estimated_source.entropy()
        print("Fuente afín")
        print(memoryless_source)
        print(f"Entropía de la fuente afín: {memoryless_source.entropy():.3g} bits")
        print("Fuente estimada")
        print(estimated_source)
        print(f"Entropía estimada de la fuente afín: {estimated_entropy:.3g}")

if __name__ == "__main__":
    entropy_sim()
    typical_set_sim()
    multicharacter_symbols_sim()
    text_source_sim()
    memory_source_sim()
    memoryless_source_sim()
    memory_typical_set_sim()
    compression_sim()
    entropy_rate_sim()
    pass
    