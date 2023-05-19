# +
import pandas as pd
import numpy as np

# Read CSV File With Pandas
df = pd.read_csv(
        "dataset-HAR-PUC-Rio.csv",
        names=["User","Gender","Age","Height","Weight","BMI","x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4","Class"],
        delimiter=';',
        #Line 0 contains information about the columns and line 122078 has a mistake
        skiprows=[0,122077],
        decimal=','
)

sensors_df = df[["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "x4", "y4", "z4", "Class"]].copy()
sensors_df.describe()
# -

from sklearn.preprocessing import MinMaxScaler 
grouped_sensor_data = sensors_df.groupby("Class")
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
#def fitness_function(solution):
    c = 0.001
    result = 0
    for i in range(1,4):
        result += cosine_similarity([np.array(solution)], [normalized_mean_data[i]])
    result = (cosine_similarity([np.array(solution)],[normalized_mean_data[0]]) + c*(1 - 0.25*result)) / (1+c)
    return result[0][0]

#print(fitness_function(normalized_mean_data[0]))


# -

import pygad
ga_instance = pygad.GA(num_generations=1000,
                       num_parents_mating=int(0.6*len(normalized_mean_data[0])),
                       fitness_func=fitness_function,
                       sol_per_pop=20,
                       num_genes=len(normalized_mean_data[0]),
                       init_range_low=0,
                       init_range_high=1,
                       parent_selection_type="sss",
                       keep_parents=1,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=1,
                      )

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Values of the best solution:\n{inverseTransformAndReshape(np.array(solution))}")
print(f"Original mean values for sitting class:\n{inverseTransformAndReshape(normalized_mean_data[0])}")
print(f"Fitness value of the best solution: {solution_fitness}")
