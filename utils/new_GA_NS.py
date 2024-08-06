import pandas as pd
import random
import numpy as np

MUTATION_PROBABILITY = 0.1
FIXED_BIAS = False
BOUNDS = [0.01, 0.99]
#~ BOUNDS = [-1, 1]
ELITE_PART = 0.4	## 0.4
range_factor = 0.3

## For testing
#~ random.seed(27)

def create_random_parameters_set(pop_size, geno_size, weights_bias_range):
    """
        Create random parameters set.
        :param pop_size: Population size
        :param geno_size: Genotype size
        :param weights_bias_range: Range with the possible values
        :return population: List of list with the genotypes
    """
    population = [0] * pop_size
    for pop_ind in range(pop_size):
        genotype = [0] * geno_size
        for ind in range(len(genotype)):
            if FIXED_BIAS:
                genotype[ind] = random.choice(weights_bias_range)
                genotype[136] = 5
                genotype[137] = 5
            else:
                genotype[ind] = random.choice(weights_bias_range)
                #~ genotype[ind] = round(random.choice(weights_bias_range), 2)
        population[pop_ind] = genotype
    return population

def create_random_single_parameters_set(weights_bias_range, geno_size):
    """
        Create random parameter set.
        :param weights_bias_range: Range with the possible values
        :param geno_size: Genotype size
        :return genotype: List of list with the genotypes
    """

    genotype = [0] * geno_size
    for ind in range(len(genotype)):
        genotype[ind] = random.choice(weights_bias_range)

    return genotype

def random_number_close_range(current_value, x, bounds, int_value = False):
    """
        Take the current value, add and subtract a constant value x to generate a randon
        value in the range(current_value - x, current_value + x)
        :param current_value: Current value that is the set to be the main value in the range
        :param x: Value to add and subtract to current_value

        :return: Random number in range(current_value - x, current_value + x)
    """

    lower_bound = current_value - x
    upper_bound = current_value + x

    # if lower_bound < 0:
    #     lower_bound = 0

    if int_value:
        ## Generate random integer number between lower and upper bounds
        random_rational_number = random.randint(lower_bound, upper_bound)

    else:
        ## Generate random rational number between lower and upper bounds
        close_range = np.arange(lower_bound, upper_bound, 0.01)
        #~ print("Close range value: ", close_range)
        random_rational_number = random.choice(close_range)
        while random_rational_number < bounds[0] or random_rational_number > bounds[1]:
            random_rational_number = random.choice(close_range)
        #~ print(f"Random rational number: {random_rational_number}")
        
    return random_rational_number

def population_reproduce_novelty(novelty_archive, p, pop_size, n_genes):
    """
        Create new population based on the novelty.
        :param p: Population - List.
        :param novelty: Population novelty - List.
        :param n_genes: Number of genes in my genotype - Int.
    """

    new_p = []

    ## Sort the list by 'novelty' key in descending order
    sorted_genomes = sorted(novelty_archive, key=lambda x: x['novelty'], reverse=True)

    ## Extract just the genome identifiers in sorted order
    sorted_parameters = [genome['genome'] for genome in sorted_genomes]

    #~ novelty_values = [item['novelty'] for item in sorted_genomes]
    #~ print(f"Novelty values sorted: {novelty_values}")

    ## Selection
    #~ elite_part = len(novelty_archive)
    ## I am adding just the elite part of the novelty archive
    ## Based on their novelty
    elite_part = round(len(novelty_archive) * ELITE_PART)
    new_p = new_p + sorted_parameters[:elite_part]

    #~ print(f"Elite part len: {elite_part}")

    for i in range(pop_size-elite_part):
        mom = p[random.randint(0, pop_size - 1)]
        dad = p[random.randint(0, pop_size - 1)]
        child = crossover(mom, dad)
        child = mutate(child, n_genes, FIXED_BIAS)
        new_p.append(child)
        
    #~ print(f"New pop len: {len(new_p)}")

    return new_p

def population_get_fittest(p,f):

    f = np.array(f)
    p = np.array(p)

    pop_best_fitness = max(f)
    # print("max fitness", pop_best_fitness)
    pop_best_fitness_pos = np.argmax(f)

    # print("pos best fitness position: ", pop_best_fitness_pos)

    pop_best_params = p[pop_best_fitness_pos]
    # print("best params: ", pop_best_params)

    return pop_best_params, pop_best_fitness

def population_get_average_fitness(f):
    """
        Get population average fitness.
        :param f: List with population fitness.
    """
    return sum(f)/len(f)

def crossover(p1,p2):

    crossover = []
    locii = [random.randint(0,8) for _ in range(len(p1))]

    for i in range(len(p1)):
        if locii[i]>4:
            crossover.append(p2[i])
        else:
            crossover.append(p1[i])

    if FIXED_BIAS:
        crossover[136] = 5
        crossover[137] = 5

    return crossover

def mutate(child, n_genes, FIXED_BIAS):
    """
        Implement mutation.
    """

    if FIXED_BIAS:

        ## Set bias values to 2
        child[136] = 5
        child[137] = 5

        for gene_no in range(n_genes):
            if np.random.rand() < MUTATION_PROBABILITY:
                ## Gene 136 is bias for left motor
                ## Gene 137 is bias for right motor
                if gene_no == 136 or gene_no == 137:
                    child[gene_no] = child[gene_no]
                else:
                    child[gene_no] = random_number_close_range(child[gene_no], range_factor, BOUNDS)
    else:
        for gene_no in range(n_genes):
            if np.random.rand() < MUTATION_PROBABILITY:
                child[gene_no] = random_number_close_range(child[gene_no], range_factor, BOUNDS)
                
        #~ print(f"I am mutating")

    return child
