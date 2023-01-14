# FF:
#   FF1: maximize diversity on canvas
#   FF2: minimize size of generated code

# Note: at present I do not delete the Image object when creating a child as it
# seems to be generating more "interesting" outputs than if I were to replace the
# Image attribute with a blank canvas.

from PIL import Image, ImageDraw, ImageChops
import opensimplex
import random
import tracery
import math
import os
from itertools import repeat
from generative_object import GenerativeObject
from techniques import *
from copy import deepcopy
import multiprocessing as mpc
import numpy as np
import argparse
from settings import *

parser = argparse.ArgumentParser()
parser.add_argument('--generations', default=25, type=int)
parser.add_argument('--population_size', default=50, type=int)
parser.add_argument('--crossover_rate', default=0.4, type=float)
parser.add_argument('--mutation_rate', default=0.2, type=float)
parser.add_argument('--random_eval', action='store_true', default=False)
parser.add_argument('--output_dir', default='.')
args = parser.parse_args()


# Accepts a GenerativeObject and iterates over its grammar, performing the technique specified
def evaluate(g):  #id, dim, grammar):
    print("Evaluating {0}:{1}".format(g.id, g.grammar))
    for technique in g.grammar.split(','):
        _technique = technique.split(":")  # split off parameters
        c = (random.randint(0,255), random.randint(0,255), random.randint(0, 255))
        if _technique[0] == 'flow-field':
            flowField(g.image, 1, g.dim[1], g.dim[0], c, _technique[1],
                      _technique[2], _technique[2])
        elif _technique[0] == 'stippled':
            stippledBG(g.image, c, g.dim)
        elif _technique[0] == 'pixel-sort':
            # 1: angle
            # 2: interval
            # 3: sorting algorithm
            # 4: randomness
            # 5: character length
            # 6: lower threshold
            # 7: upper threshold
            g.image = pixelSort(g.image, _technique[1:])

        elif _technique[0] == 'dither':
            g.image = simpleDither(g.image)
        elif _technique[0] == 'wolfram-ca':
            WolframCA(g.image)
        elif _technique[0] == 'drunkardsWalk':
            drunkardsWalk(g.image)
        elif _technique[0] == 'flow-field-2':
            flowField2(g.image, _technique[1], _technique[2], _technique[3], _technique[4])
        elif _technique[0] == 'circle-packing':
            circlePacking(g.image, _technique[1], _technique[2])

    return g


# Evaluate all 'unevaluated' members of the current population
def evaluatePopulation(_population):
    unevaluated = list(filter(lambda x: not x.isEvaluated, _population))
    with mpc.Pool(mpc.cpu_count() - 3) as p:
        # with mpc.Pool(4) as p:
        retval = p.starmap(evaluate, zip(unevaluated))
        for i in range(len(retval)):
            assert unevaluated[i].id == retval[
                i].id, "Error with ID match on re-joining."
            unevaluated[i].isEvaluated = True
            unevaluated[i].image = retval[i].image


# Fill in the passed in list with random population members up to the population_size parameter
def generatePopulation(_population, gen, pop_size):
    ret_pop = _population.copy()
    print("Generating new population from size {0} to size {1}".format(
        len(_population), pop_size))
    i = 0
    while len(ret_pop) < pop_size:
        idx = "{0}_{1}".format(str(gen), i)
        #g = GenerativeObject(idx, DIM, grammar.flatten("#ordered_pattern#"))
        g = GenerativeObject(DIM, grammar.flatten("#ordered_pattern#"), background='black', idx=idx)
        ret_pop.append(g)
        i += 1
    return ret_pop


# Compare each population member to each other population member (this one uses RMS difference)
# and set its fitness to be the greatest 'difference'
def pairwiseComparison(_population):
    maxUniques = 0
    maxDiff = 0.0
    compared = {}
    for p in _population:
        maxUniques = len(set(p.grammar.split(',')))
        psum = 0

        # image is the background color with no changes - weed out
        numblack = count_nonblack_pil(p.image)
        if numblack == 0:
            p.setFitness(-1.0)
        else:
            for p2 in _population:
                if p != p2:
                    maxUniques2 = len(set(p2.grammar.split(',')))
                    if maxUniques2 > maxUniques:
                        maxUniques = maxUniques2

                    id1 = "{0}:{1}".format(p.id, p2.id)
                    id2 = "{0}:{1}".format(p2.id, p.id)
                    keys = compared.keys()
                    if not id1 in keys or not id2 in keys:
                        diff = rmsdiff(p.image, p2.image)

                        if (diff > maxDiff):
                            maxDiff = diff
                        compared[id1] = True
                        psum += diff
            psum /= (len(_population) - 1)
            p.setFitness(psum)

    # actual fitness?
    for p in _population:
        lenTechniques = len(set(p.grammar.split(',')))
        p.setFitness((0.5 * (p.getFitness() / maxDiff)) +
                     (0.5 * (lenTechniques / maxUniques)))


# Perform single-point crossover
def singlePointCrossover(_population, _next_pop, num_xover):
    for j in range(int(num_xover / 2)):
        id1 = random.randint(0, len(_population) - 1)
        id2 = random.randint(0, len(_population) - 1)
        while id1 == id2:
            id2 = random.randint(0, len(_population) - 1)

        # children
        c1 = deepcopy(_population[id1])
        c2 = deepcopy(_population[id2])

        c1.isEvaluated = False
        c2.isEvaluated = False
        c1.id += "_c_{0}1_g{1}".format(j, gen)
        c2.id += "_c_{0}2_g{1}".format(j, gen)

        split_grammar1 = _population[id1].grammar.split(",")
        split_grammar2 = _population[id2].grammar.split(",")

        if len(split_grammar1) > 1 and len(split_grammar2) > 1:
            # crossover for variable length
            # pick an index each and flop
            xover_idx1 = random.randint(1, len(split_grammar1) - 1)
            xover_idx2 = random.randint(1, len(split_grammar2) - 1)

            new_grammar1 = []
            new_grammar2 = []

            print(len(split_grammar1), len(split_grammar2), xover_idx1,
                  xover_idx2)
            # up to indices
            for i in range(xover_idx1):
                new_grammar1.append(split_grammar1[i])
            for i in range(xover_idx2):
                new_grammar2.append(split_grammar2[i])

            # past indices
            for i in range(xover_idx1, len(split_grammar1)):
                new_grammar2.append(split_grammar1[i])
            for i in range(xover_idx2, len(split_grammar2)):
                new_grammar1.append(split_grammar2[i])

        else:  # one of the genomes was length 1
            new_grammar1 = []
            new_grammar2 = []

            if len(split_grammar1) == 1:
                new_grammar2 = split_grammar2.copy()
                new_grammar2.insert(random.randint(0, len(split_grammar2)),
                                    split_grammar1[0])

                new_grammar1 = split_grammar2.copy()
                new_grammar1.insert(random.randint(0, len(split_grammar2)),
                                    split_grammar1[0])
            else:
                new_grammar2 = split_grammar1.copy()
                new_grammar2.insert(random.randint(0, len(split_grammar1)),
                                    split_grammar2[0])

                new_grammar1 = split_grammar1.copy()
                new_grammar1.insert(random.randint(0, len(split_grammar1)),
                                    split_grammar2[0])

        c1.grammar = ",".join(new_grammar1)
        c2.grammar = ",".join(new_grammar2)
        _next_pop.append(c1)
        _next_pop.append(c2)

        print("---")
        print(c1.id, c1.grammar, _population[id1].id, _population[id1].grammar)
        print(c2.id, c2.grammar, _population[id2].id, _population[id2].grammar)
        print("---")


# And single-point mutation
def singlePointMutation(_population, _next_pop, num_mut):
    for j in range(num_mut):
        pop_id = random.randint(0, len(_population) - 1)
        mutator = deepcopy(_population[pop_id])
        mutator.id += "_m_{0}_g{1}".format(j, gen)
        #mutator.image = Image.new("RGBA", DIM, "black")
        # leaving the 'old' image makes it look neater imo...
        mutator.isEvaluated = False

        split_grammar = mutator.grammar.split(",")
        mut_idx = random.randint(0, len(split_grammar) - 1)
        local_grammar = grammar.flatten("#technique#")
        split_grammar[mut_idx] = local_grammar
        mutator.grammar = ",".join(split_grammar)

        _next_pop.append(mutator)


if __name__ == "__main__":
    opensimplex.seed(random.randint(0, 100000))

    # pull in cmd-line parameters
    num_gens = args.generations
    pop_size = args.population_size

    # create output dir if it doesn't exist
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)

    xover_rate = args.crossover_rate
    mut_rate = args.mutation_rate
    population = []

    if args.random_eval:  # random
        run_type = "random"
        pop_size = num_gens * pop_size
        print("Random evaluation")
        population = generatePopulation(population, 0, pop_size)

        # initial evaluation
        evaluatePopulation(population)

        # pair-wise comparison
        pairwiseComparison(population)

        population.sort(key=lambda x: x.fitness_internal, reverse=True)
    else:  # ga
        run_type = "ga"
        ##### GENERATION 0
        print("Generation", 0)
        population = generatePopulation(population, 0, pop_size)

        # initial evaluation
        evaluatePopulation(population)

        # pair-wise comparison
        pairwiseComparison(population)

        population.sort(key=lambda x: x.fitness_internal, reverse=True)
        print("Generation {0} best fitness: {1}, {2}, {3}".format(
            0, population[0].fitness_internal, population[0].grammar, population[0].id))
        print("---")
        #####################

        for gen in range(1, num_gens):
            print("Generation", gen)

            num_xover = int(pop_size * xover_rate)
            num_mut = int(pop_size * mut_rate)
            next_pop = []

            next_pop.append(deepcopy(population[0]))  # elite

            # crossover
            singlePointCrossover(population, next_pop, num_xover)

            # mutation
            singlePointMutation(population, next_pop, num_mut)

            # filling in
            next_pop = generatePopulation(next_pop, gen, pop_size)

            # evaluation
            evaluatePopulation(next_pop)

            # pair-wise comparison
            pairwiseComparison(next_pop)

            # Sorting and cleanup
            next_pop.sort(key=lambda x: x.fitness_internal, reverse=True)
            print("Generation {0} best fitness: {1}, {2}, {3}".format(
                gen, population[0].fitness_internal, population[0].grammar,
                population[0].id))
            print("---")
            del population
            population = next_pop

        # Final evaluation
        evaluatePopulation(population)
        pairwiseComparison(population)
        population.sort(key=lambda x: x.fitness_internal, reverse=True)

    # Print out last generation
    print("Final output:")

    for i in range(len(population)):
        print(population[i].id, population[i].fitness_internal, population[i].grammar)
        if i == 0:
            population[i].image.save(
                os.path.join(
                    args.output_dir,
                    "best-img-{0}.{1}.png".format(population[i].id, run_type)))
        else:
            population[i].image.save(
                os.path.join(
                    args.output_dir,
                    "img-{0}.{1}.png".format(population[i].id, run_type)))
    print("---")
    print("End of line.")
