# +
import pandas as pd
import numpy as np

# Read CSV File With Pandas
df = pd.read_csv(
        "dataset-HAR-PUC-Rio.csv",
        delimiter=';',
        #Line 0 contains information about the columns and line 122078 has a mistake
        skiprows=[122077],
        decimal=','
)

#Keep only columns I need
sensors_df = df[["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "x4", "y4", "z4", "class"]].copy()
sensors_df.describe()

# +
from sklearn.preprocessing import MinMaxScaler 

#Aggregate the data
grouped_sensor_data = sensors_df.groupby("class")
mean_sensor_data = grouped_sensor_data.agg({
    "x1":'mean',
    "y1":'mean',
    "z1":'mean',
    "x2":'mean',
    "y2":'mean',
    "z2":'mean',
    "x3":'mean',
    "y3":'mean',
    "z3":'mean',
    "x4":'mean',
    "y4":'mean',
    "z4":'mean'
})

#MinMax Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
normalized_mean_data = scaler.fit_transform(mean_sensor_data)

#Function to inverse transform the data when necessary
def inverseTransformAndReshape(array):
    return scaler.inverse_transform(np.reshape(array, (1,-1)))


# +
from sklearn.metrics.pairwise import cosine_similarity

c = 0.4

#Fitness function to use with GA
def fitness_function(ga_instance, solution, solution_idx):
    result = 0
    for i in range(1,4):
        result += cosine_similarity([np.array(solution)], [normalized_mean_data[i]])
    result = (cosine_similarity([np.array(solution)],[normalized_mean_data[0]]) + c*(1 - 0.25*result)) / (1+c)
    return result[0][0]

#Used to manually execute the fitness funtion
def manual_fitness_function(solution):
    result = 0
    for i in range(1,4):
        result += cosine_similarity([np.array(solution)], [normalized_mean_data[i]])
    result = (cosine_similarity([np.array(solution)],[normalized_mean_data[0]]) + c*(1 - 0.25*result)) / (1+c)
    return result[0][0]


# +
best_fitness_values = []
num_of_insignificant_better = []
num_of_insignificant_better.append(0)

#Stop early when GA converges
def early_stopping_callback(ga_instance):
    best_fitness_values.append(ga_instance.best_solution()[1])
        
    better_ratio = (best_fitness_values[len(best_fitness_values)-1] / best_fitness_values[len(best_fitness_values)-2]) - 1
    if better_ratio < 0.01 and len(best_fitness_values) > 1:
        num_of_insignificant_better.append(num_of_insignificant_better[len(num_of_insignificant_better)-1] + 1)
    else:
        num_of_insignificant_better.append(0)
    
    if num_of_insignificant_better[len(num_of_insignificant_better)-1] == 100:
        return "stop"


# +
from scipy.stats import qmc

initial_pop = []

number_generations = 1000
population_size = 20
crossover_chance = 0.6
mutation_chance = 0.01

#Create initial populations
for i in range(0,10):
    sampler = qmc.LatinHypercube(d=12)
    initial_pop.append(sampler.random(n=population_size))

# +
import pygad

ga_instances = []

#Define GA instances
for i in range(0,10):
    ga_instance = pygad.GA(initial_population=initial_pop[i], #Initial Population with Latin Hypercube Sampling
                       gene_space={'low':0, 'high':1},
                       num_generations=number_generations,
                       crossover_probability=crossover_chance,
                       num_parents_mating = population_size,
                       fitness_func=fitness_function,
                       #sol_per_pop=population_size, #For randomized initialization of initial population
                       #num_genes=len(normalized_mean_data[0]), #For randomized initialization of initial population
                       #init_range_low=0,
                       #init_range_high=1,
                       parent_selection_type="rws",
                       keep_elitism=1,
                       crossover_type="two_points",
                       mutation_type="random",
                       mutation_probability=mutation_chance,
                       on_generation=early_stopping_callback
                      )
    ga_instances.append(ga_instance)

# +
best_fitness_per_generation_per_execution = []
number_of_generations_per_execution = []
best_values_per_execution = []
best_fitness_per_execution = []

#Print results for every instance
print(f"Original mean values for 'sitting' class:\n{inverseTransformAndReshape(normalized_mean_data[0])}")
print(f"Fitness value of original 'sitting' means: {manual_fitness_function(normalized_mean_data[0])}\n")

for i in range(0,10):
    print(f"Execution number {i+1}:")
    ga_instances[i].run()
    solution, solution_fitness, solution_idx = ga_instances[i].best_solution()
    print(f"Early Stopping on generation {len(best_fitness_values)}")
    print(f"Values of the best solution:\n{inverseTransformAndReshape(np.array(solution))}")
    print(f"Fitness value of the best solution: {solution_fitness}\n")
    best_fitness_per_generation_per_execution.append(np.pad(np.array(best_fitness_values),
        (0, number_generations - len(best_fitness_values)), mode='constant', constant_values=np.nan))
    number_of_generations_per_execution.append(len(best_fitness_values))
    best_values_per_execution.append(np.reshape(inverseTransformAndReshape(np.array(solution)), (12)))
    best_fitness_per_execution.append(solution_fitness)
    best_fitness_values = []
    num_of_insignificant_better = []
    
print(f"Mean number of generations for all executions: {np.array(number_of_generations_per_execution).mean()}")
print(f"Mean values of best solution for all executions: {np.array(best_values_per_execution).mean(axis=1)}")
print(f"Mean fitness of best solution for all executions: {np.array(best_fitness_per_execution).mean()}")

# +
from matplotlib import pyplot as plt

#Show graphs for every instance
best_fitness_per_generation_means = np.array(best_fitness_per_generation_per_execution)
best_fitness_per_generation_means = np.nanmean(best_fitness_per_generation_means, axis=0)

plt.plot(best_fitness_per_generation_means)
plt.legend(['Mean Best Chromosome of all Executions',], loc='best')
plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.title(f"Population Size = {population_size}, Crossover Chance = {crossover_chance}, Mutation Chance = {mutation_chance}")
plt.show()

plt.plot(np.array(best_fitness_per_generation_per_execution)[0])
plt.plot(np.array(best_fitness_per_generation_per_execution)[1])
plt.plot(np.array(best_fitness_per_generation_per_execution)[2])
plt.plot(np.array(best_fitness_per_generation_per_execution)[3])
plt.plot(np.array(best_fitness_per_generation_per_execution)[4])
plt.plot(np.array(best_fitness_per_generation_per_execution)[5])
plt.plot(np.array(best_fitness_per_generation_per_execution)[6])
plt.plot(np.array(best_fitness_per_generation_per_execution)[7])
plt.plot(np.array(best_fitness_per_generation_per_execution)[8])
plt.plot(np.array(best_fitness_per_generation_per_execution)[9])
plt.legend(['Best Chromosome in Execution 1',
            'Best Chromosome in Execution 2',
            'Best Chromosome in Execution 3',
            'Best Chromosome in Execution 4',
            'Best Chromosome in Execution 5',
            'Best Chromosome in Execution 6',
            'Best Chromosome in Execution 7',
            'Best Chromosome in Execution 8',
            'Best Chromosome in Execution 9'
           ], loc='best')
plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.title(f"Population Size = {population_size}, Crossover Chance = {crossover_chance}, Mutation Chance = {mutation_chance}")
plt.show()
