def tests():
    dist = {
        "0": 0.1,
        "1": 0.8,
    }

    extended_dist = dict()
    aux_dist = dict()
    for simbol, probability in dist.items():
        extended_dist[simbol] = probability
        
    n = 15
    for i in range(n - 1):
        aux_dist = dict()
        for extended_simbol, extended_probability in extended_dist.items():
            for simbol, probability in dist.items():
                aux_dist[extended_simbol + simbol] = extended_probability*probability
        extended_dist = dict()
        for aux_simbol, aux_probability in aux_dist.items():
            extended_dist[aux_simbol] = aux_probability
    print(len(extended_dist))
    
    for i in range(n + 1):
        d = dict()
        for j in range(len(dist)**i):
            d[j] = j
    
    print(len(d))


if __name__ == "__main__":
    tests()
    