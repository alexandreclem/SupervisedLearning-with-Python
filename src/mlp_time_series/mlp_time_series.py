# Data retrieved using Google Cloud BigQuery python API
# Dataset name: covid19_open_data
# Dataset URL: https://console.cloud.google.com/marketplace/details/bigquery-public-datasets/covid19-public-data-program?project=ethereum-public-data
# Query used: 
'''
SELECT date, location_key, new_deceased, new_confirmed, cumulative_deceased, cumulative_tested, new_persons_vaccinated, cumulative_persons_fully_vaccinated, new_vaccine_doses_administered, cumulative_vaccine_doses_administered, new_hospitalized_patients, new_intensive_care_patients, cumulative_hospitalized_patients, cumulative_intensive_care_patients
FROM `bigquery-public-data.covid19_open_data.covid19_open_data` WHERE location_key='BR' AND date BETWEEN '2022-01-01' AND '2022-06-01' ORDER BY date
'''


import random
import numpy as np

# Visualizations
from matplotlib import pyplot as plt

# Preparation
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ML Model
import tensorflow.random as tfr
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_squared_error


def sequential_dataset(x, y, time_steps):    
    new_x = []
    new_y = []
    limit = x.shape[0] - time_steps
    for i in range(limit):
        x_samples = x[i: i + time_steps, :]
        y_samples = y[i + time_steps, :]
        new_x.append(x_samples)
        new_y.append(y_samples)    
    new_x = np.array(new_x, dtype='float64')
    new_y = np.array(new_y, dtype='float64')
    
    return new_x, new_y


def main():
    # Seeds
    random.seed(10)
    np.random.seed(10)
    tfr.set_seed(10)

    # Optimal Configuration #
    k = 1 # time steps
    val_percent = 0.1
    hidden_neurons = 95 # only one hidden layer, type: lstm (long short-term memory)
    output_neurons = 1 # type: dense (fully connected)
    lr = 0.025 # learning rate
    hidden_activation_function = 'tanh'
    output_activation_function = 'tanh'
    performance_function = 'mean_squared_error'
    opt = Adam(learning_rate=lr) # optimizer    
    max_epochs = 150
    max_val_fail = 75 # early stopping, validation fails   
    
    # Loading Data #    
    training_df = pd.read_csv('training.csv')
    testing_df = pd.read_csv('testing.csv')
    data = pd.concat([training_df, testing_df], axis=0) # Complete dataset 

    # Preparation #
    # Treating NaN values    
    data = data.drop('cumulative_tested', axis=1) # Removing this column due to a lot of NaN values
    data = data.fillna(value=int(testing_df['new_hospitalized_patients'].mean())) # Filling with the column mean
    data = data.fillna(value=int(testing_df['new_intensive_care_patients'].mean()))
    data = data.fillna(value=int(testing_df['cumulative_hospitalized_patients'].mean()))
    data = data.fillna(value=int(testing_df['cumulative_intensive_care_patients'].mean()))       
    
    # Pre-processing #    
    # Adjusting the data    
    targets = data[['new_deceased']].copy()               
    data = data[['new_deceased']].copy() # Using only new_deceased as feature
    # data = data.drop(['date', 'location_key'], axis=1) # Using all features
    targets = targets.to_numpy()
    data = data.to_numpy()

    # Scaling the data       
    data_scaler = MinMaxScaler(feature_range=(0, 1)) # Normalizing the input (set)
    data_scaled = data_scaler.fit_transform(data)      

    target_scaler = MinMaxScaler(feature_range=(0, 1)) # Normalizing the output (set)
    targets_scaled = target_scaler.fit_transform(targets)   

    # Dividing the data    
    training_set = []
    training_targets = []
    testing_set = []
    testing_targets = []
    
    training_range = data.shape[0] - 31 # January to May 2022   
    for i in range(training_range):
        training_set.append(data_scaled[i, :])
        training_targets.append(targets_scaled[i, :])

    training_set = np.array(training_set, dtype='float64')
    training_targets = np.array(training_targets, dtype='float64')

    for i in range(training_range, data.shape[0]):
        testing_set.append(data_scaled[i, :])
        testing_targets.append(targets_scaled[i, :])

    testing_set = np.array(testing_set, dtype='float64')
    testing_targets = np.array(testing_targets, dtype='float64')
        
    # Creating the sequences
    time_steps = k
    training_set_seq, training_targets_seq = sequential_dataset(training_set, training_targets, time_steps)
    testing_set_seq, testing_targets_seq = sequential_dataset(testing_set, testing_targets, time_steps)   
    
    # Topology/Hyperparameters and Training #    
    model = Sequential()   
    
    model.add( 
        Bidirectional(
            LSTM(
                units=hidden_neurons,
                activation=hidden_activation_function,                                               
                input_shape=(training_set_seq.shape[1],
                training_set_seq.shape[2])
            )
        )
    )   
   
    model.add(
        Dense(
            units=output_neurons,                      
            activation=output_activation_function
        )
    )        
    
    model.compile(
        optimizer=opt,
        loss=performance_function            
    )    
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=max_val_fail, verbose=0)
    history = model.fit(
        x=training_set_seq,
        y=training_targets_seq,
        validation_split=val_percent,        
        epochs=max_epochs,
        shuffle=False,
        verbose=1,
        callbacks=[early_stopping]      
    )

    # Testing and Evaluating #
    predictions_scaled = model.predict(x=testing_set_seq, verbose=0)
    predictions = target_scaler.inverse_transform(predictions_scaled)    
    true_targets = target_scaler.inverse_transform(testing_targets)
    true_targets = true_targets[0: testing_targets.shape[0] - time_steps, :]
    days = [i + 1 for i in range(testing_targets.shape[0] - time_steps)] 
    
    error = []            
    for i in range(len(days)):
        error.append(mean_squared_error([true_targets[i, 0]], [predictions[i, 0]], squared=False))                
    mean_error = sum(error) / len(error) 
    mean_error = [mean_error for i in range(len(days))]     
    
    print(f'\nMean RMSE error: {round(mean_error[0], 2)}')    

    training_error = history.history['loss']
    validation_error = history.history['val_loss']  
    epochs = [i for i in range(len(training_error))]      

    # Training and Validation error (MSE) x Epoch
    plt.figure(3)     

    plt.plot(epochs, training_error, color='r', linestyle='solid', linewidth=1, label='training_error')      
    plt.plot(epochs, validation_error, color='black', linestyle='solid', linewidth=1, label='validation_error')      

    plt.title('Error (MSE) per Epoch', loc='center', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('error', fontsize=15)
    plt.axis('tight')
    plt.legend(loc='upper left', prop={'size': 9})
    
    # MSE x Days
    plt.figure(2)    
    
    plt.plot(days, error, color='r', linestyle='solid', linewidth=1, label='error')          
    plt.plot(days, mean_error, color='black', linestyle='solid', linewidth=1, label='mean_error')              

    plt.title('Error (RMSE) per Day (May 2022)')
    plt.xlabel('day')    
    plt.ylabel('error')       
    plt.axis('tight')
    plt.legend(loc='upper left', prop={'size': 9})
    
    # True Values x Predictions
    plt.figure(1) 

    plt.plot(days, true_targets, color='g', linestyle='solid', linewidth=1, label='true')      
    plt.plot(days, predictions, color='b', linestyle='solid', linewidth=1, label='predictions')      

    plt.title('Deaths per Day (May 2022)', loc='center', fontsize=15)
    plt.xlabel('day', fontsize=15)
    plt.ylabel('deaths', fontsize=15)
    plt.axis('tight')
    plt.legend(loc='upper left', prop={'size': 9})

    plt.tight_layout()  
    plt.show()  

    
if __name__ == '__main__':
    main()