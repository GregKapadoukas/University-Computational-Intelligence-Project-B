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

sensors_df = df[["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "x4", "y4", "z4", "class"]].copy()
sensors_df.describe()
# -

from sklearn.preprocessing import MinMaxScaler 
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
scaler = MinMaxScaler(feature_range=(0,1))
normalized_mean_data = scaler.fit_transform(mean_sensor_data)

# +
from sklearn.metrics.pairwise import cosine_similarity

def inverseTransformAndReshape(array):
    return scaler.inverse_transform(np.reshape(array, (1,-1)))

def fitness_function(ga_instance, solution, solution_idx):
#def fitness_function(solution): # for manually running and testing
    c = 0.2
    result = 0
    for i in range(1,4):
        result += cosine_similarity([np.array(solution)], [normalized_mean_data[i]])
    result = (cosine_similarity([np.array(solution)],[normalized_mean_data[0]]) + c*(1 - 0.25*result)) / (1+c)
    return result[0][0]

#print(fitness_function(normalized_mean_data[0]))


# -

best_fitness_values = []
num_of_insignificant_better = []
num_of_insignificant_better.append(0)
def early_stopping_callback(ga_instance):
    best_fitness_values.append(ga_instance.best_solution()[1])
        
    better_ratio = (best_fitness_values[len(best_fitness_values)-1] / best_fitness_values[len(best_fitness_values)-2]) - 1
    #print(best_fitness_values[len(best_fitness_values)-1])
    #print(best_fitness_values[len(best_fitness_values)-2])
    #print(better_ratio)
    #print(num_of_insignificant_better[len(num_of_insignificant_better)-1])
    #print("\n")
    if better_ratio < 0.01 and len(best_fitness_values) > 1:
        num_of_insignificant_better.append(num_of_insignificant_better[len(num_of_insignificant_better)-1] + 1)
    else:
        num_of_insignificant_better.append(0)
    
    if num_of_insignificant_better[len(num_of_insignificant_better)-1] == 50:
        return "stop"


# +
from scipy.stats import qmc

sampler = qmc.LatinHypercube(d=12)
initial_pop = sampler.random(n=20)
# -

ga_instance = pygad.GA(initial_population=initial_pop, #Initial Population with Latin Hypercube Sampling
                       num_generations=1000,
                       num_parents_mating=int(0.6*len(normalized_mean_data[0])),
                       fitness_func=fitness_function,
                       #sol_per_pop=20, #For randomized initialization of initial population
                       #num_genes=len(normalized_mean_data[0]), #For randomized initialization of initial population
                       init_range_low=0,
                       init_range_high=1,
                       parent_selection_type="sss",
                       keep_parents=1,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=1,
                       on_generation=early_stopping_callback
                      )
#ga_instance.run_callbacks.append(early_stopping_callback)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Values of the best solution:\n{inverseTransformAndReshape(np.array(solution))}")
print(f"Original mean values for sitting class:\n{inverseTransformAndReshape(normalized_mean_data[0])}")
print(f"Fitness value of the best solution: {solution_fitness}")
ga_instance.plot_fitness()

ga_instance.initial_population.shape
