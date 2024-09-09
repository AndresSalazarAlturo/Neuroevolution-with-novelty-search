import pandas as pd
import random
import numpy as np

MUTATION_PROBABILITY = 0.1
BOUNDS = [0.01, 0.99]
#~ BOUNDS = [-1, 1]
range_factor = 0.3
tournament_size = 2
num_tournaments = 60

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
    
def tournament_selection(novelty_archive):
    """
        Perform tournament selection.
        :param novelty_archive: Population novelty - list
        :param tournament_size: Size of the tournament - int
        :return: Winner and loser genotypes
    """
    
    candidate1, candidate2 = random.sample(novelty_archive, tournament_size)
    
    if candidate1['novelty'] > candidate2['novelty']:
        winner = candidate1
        loser = candidate2
    else:
        winner = candidate2
        loser = candidate1	
    
    return winner

def population_reproduce_novelty(novelty_archive, p, pop_size, n_genes):
    """
        Create new population based on the novelty.
        :param p: Population - List.
        :param novelty: Population novelty - List.
        :param n_genes: Number of genes in my genotype - Int.
    """

    new_p = []

    ## Tournament selection
    for _ in range(num_tournaments):

        winner = tournament_selection(novelty_archive)
        new_p.append(winner['genome'])
        
    ## Testing reproduce from the novelty archive
    p = [genome["genome"] for genome in novelty_archive]

    for i in range(pop_size-num_tournaments):
		
        #~ mom = p[random.randint(0, pop_size - 1)]
        #~ dad = p[random.randint(0, pop_size - 1)]
        
        mom = p[random.randint(0, len(novelty_archive) - 1)]
        dad = p[random.randint(0, len(novelty_archive) - 1)]
        
        ## Check that mom and dad are not the same individual
        if mom == dad:
            while mom == dad:
                #~ mom = p[random.randint(0, pop_size - 1)]
                #~ dad = p[random.randint(0, pop_size - 1)]
                
                mom = p[random.randint(0, len(novelty_archive) - 1)]
                dad = p[random.randint(0, len(novelty_archive) - 1)]

        child = crossover(mom, dad)
        child = mutate(child, n_genes)
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

    return crossover

def mutate(child, n_genes):
    """
        Implement mutation.
    """
    for gene_no in range(n_genes):
        if np.random.rand() < MUTATION_PROBABILITY:
            child[gene_no] = random_number_close_range(child[gene_no], range_factor, BOUNDS)

    return child
    
def import_json(file_final_novelty_archive, file_path_best_solution = None):
    """
        Import json files. Best solution and novelty archive as list of lists
        with the genomes to create the new initial population.
    """
    if file_path_best_solution != None:
        
        with open(file_path_best_solution, 'r') as file:
            best_solution_data = json.load(file)
            
        with open(file_final_novelty_archive, 'r') as file:
            novelty_archive_data = json.load(file)
            
        archive_genomes = [genome['genome'] for genome in novelty_archive_data]
        best_solution_genomes = [genome['genome'] for genome in best_solution_data]
        
        return archive_genomes, best_solution_genomes
        
def initial_pop_from_archive(archive_genomes, pop_size, n_genes, best_solution_genomes = None):
    """
        Create the initial population for the next environment based on the best solution and
        novelty archive.
        :param archive_genomes: List wih the genomes in the archive.
        :param pop_size: Population size.
        :param n_genes: Number of genes in genotype.
        :param best_solution_genomes: The best solution in the previous simulation.
        :return new_pop: New population genotypes.
    """

    new_p = []

    if best_solution_genomes != None:
        ## Add the best solution to the new pop
        new_p.append(best_solution_genomes)

        for _ in range(pop_size - 1):
            mom = archive_genomes[random.randint(0, len(archive_genomes) - 1)]
            dad = archive_genomes[random.randint(0, len(archive_genomes) - 1)]

            ## Check that mom and dad are not the same individual
            if mom == dad:
                while mom == dad:
                    mom = archive_genomes[random.randint(0, len(archive_genomes) - 1)]
                    dad = archive_genomes[random.randint(0, len(archive_genomes) - 1)]
            child = crossover(mom, dad)
            child = mutate(child, n_genes)
            new_p.append(child)

        return new_p

    else:
        for _ in range(pop_size):
            mom = archive_genomes[random.randint(0, len(archive_genomes) - 1)]
            dad = archive_genomes[random.randint(0, len(archive_genomes) - 1)]

            ## Check that mom and dad are not the same individual
            if mom == dad:
                while mom == dad:
                    mom = archive_genomes[random.randint(0, len(archive_genomes) - 1)]
                    dad = archive_genomes[random.randint(0, len(archive_genomes) - 1)]
            child = crossover(mom, dad)
            child = mutate(child, n_genes)
            new_p.append(child)

        return new_p
