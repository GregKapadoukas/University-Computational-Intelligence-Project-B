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

#Fitness funtion to use with GA
def fitness_function(ga_instance, solution, solution_idx):
    result = 0
    for i in range(1,4):
        result += cosine_similarity([np.array(solution)], [normalized_mean_data[i]])
    result = (cosine_similarity([np.array(solution)],[normalized_mean_data[0]]) + c*(1 - 0.25*result)) / (1+c)
    return result[0][0]

#Used to manually execute the fitness function
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
print(f"Mean values of best solution for all executions: {np.array(best_values_per_execution).mean(axis=0)}")
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

# +
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

#Copy best values for GAs for scaling
scaled_best_values_per_execution = np.array(best_values_per_execution)

sensor_measurements = sensors_df.copy()
sensor_classes = sensor_measurements.pop("class")

# One-hot encode output to use in the model later
sensor_classes = pd.get_dummies(df['class'])

# Standardization of input data and best GA output data
sensor_measurements = pd.concat([sensor_measurements,pd.DataFrame(scaled_best_values_per_execution, columns=list(sensor_measurements))])
for column in sensor_measurements.columns:
    sensor_measurements[column] = (sensor_measurements[column] - sensor_measurements[column].mean()) / (sensor_measurements[column].std())    
    
pandas_best_values_per_execution = sensor_measurements[165632:]
sensor_measurements = sensor_measurements[:165632]
scaled_best_values_per_execution = pandas_best_values_per_execution.to_numpy()

#Set number of neurons in hidden layers and number of max epochs
num_l1_hidden_neurons = 23
num_l2_hidden_neurons = 20
num_l3_hidden_neurons = 18
num_epochs = 100

#Join measurements and classes again in order to split for 5-fold CV
preprocessed_df = pd.concat([sensor_measurements, sensor_classes], axis=1)

#Define model
model = keras.Sequential(
    [
        keras.Input(shape=(12)),
        layers.Dense(num_l1_hidden_neurons, activation='relu'),
        layers.Dense(num_l2_hidden_neurons, activation='relu'),
        layers.Dense(num_l3_hidden_neurons, activation='relu'),
        layers.Dense(5, activation='softmax')
    ]
)
model.summary()

#Choose loss function, optimizers, learning rate, beta values and metrics
model.compile(
    loss = keras.losses.CategoricalCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate = 0.001, beta_1=0.6, beta_2=0.999),
    metrics = ["categorical_crossentropy", "mse", "categorical_accuracy"]
)

# Split data again into measurements and classes
train, validation = train_test_split(preprocessed_df, test_size=0.20, shuffle= True)

train_measurements = preprocessed_df.iloc[:,:12]
train_classes = preprocessed_df.iloc[:,12:]

validation_measurements = validation.iloc[:,:12]
validation_classes = validation.iloc[:,12:]

#Define callback for early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss')

#Train the model and evaluate
model_fit_history = model.fit(train_measurements, train_classes, batch_size=32, epochs=num_epochs, callbacks=[callback], validation_data=(validation_measurements, validation_classes), verbose=2)
model_evaluate_history = model.evaluate(validation_measurements, validation_classes, batch_size=32, verbose=2)

# +
from sklearn.preprocessing import StandardScaler 

#Find execution with higest fitness to enter in report
print(f"The execution with the highest fitness was: {best_fitness_per_execution.index(max(best_fitness_per_execution))+1}\n")

#Print NN results with GA outputs as inputs
i = 1
for result in scaled_best_values_per_execution:
    #print(model.predict(np.reshape(result, (1, -1))))
    #results_classes.append(model.predict(np.reshape(result, (1, -1))))
    print(f"Results from execution {i}")
    found_class = model.predict(np.reshape(np.array(result),(1,-1)))
    found_class = np.reshape(found_class, (5)).tolist()
    print(f"The percentage for class 'sitting' is: {found_class[0]}")
    class_index = found_class.index(max(found_class))
    if class_index == 0:
       print(f"The class with highest likelihood was 'sitting'\n") 
    elif class_index == 1:
       print(f"The class with highest likelihood was 'sittingdown' with percentage {max(found_class)}\n") 
    elif class_index == 2:
       print(f"The class with highest likelihood was 'standing' with percentage {max(found_class)}\n") 
    elif class_index == 3:
       print(f"The class with highest likelihood was 'standingup' with percentage {max(found_class)}\n") 
    elif class_index == 4:
       print(f"The class with highest likelihood was 'walking' with percentage {max(found_class)}\n") 
    i+=1
