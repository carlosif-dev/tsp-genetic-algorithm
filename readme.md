# Traveling Salesman Problem Solver

This code implements a solution for the Traveling Salesman Problem (TSP) using a genetic algorithm approach. The TSP is a classic problem in combinatorial optimization where the goal is to find the shortest possible route that visits each city exactly once and returns to the original city.

## Features

- Random City Generation: Generates a random set of cities within specified coordinate ranges.
- Genetic Algorithm: Utilizes a genetic algorithm to evolve a population of possible solutions.
- Fitness Evaluation: Calculates the total distance of a given route as the fitness value.
- Crossover and Mutation: Implements ordered crossover and swap mutation operations to create new solutions.
- Elitism: Maintains the best solution from previous generations.
- Visualization: Provides a visualization of the best solution found in each generation.

## Usage

1. Define the number of cities (num_cities) and the coordinate ranges (x_range and y_range).
1. Generate random cities using the generate_random_cities function.
1. Tune parameters such as population size (pop_size), number of generations (num_generations), and mutation rate (mutation_rate).
1. Execute the genetic_algorithm function with the appropriate arguments.
1. The best solution and its corresponding fitness value will be returned.

## Requirements

    Python 3.x
    NumPy
    Matplotlib

## Example

```
num_cities = 30
x_range = (0, 10)  # X coordinate range
y_range = (0, 10)  # Y coordinate range

cities, cities_keys = generate_random_cities(num_cities, x_range, y_range)

best_solution, best_fitness, fitness_history = genetic_algorithm(
    pop_size=1000,
    num_generations=200,
    mutation_rate=0.1,
    elements=cities_keys,
    dictionary=cities,
)


# Plot the best solution route
x = [cities[city][0] for city in best_solution + [best_solution[0]]]
y = [cities[city][1] for city in best_solution + [best_solution[0]]]
plt.plot(x, y, "o-")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Best Solution for TSP, Distance: {:.2f}".format(best_fitness))
plt.show()
```
