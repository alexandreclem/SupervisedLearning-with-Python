import sys
import random
import numpy as np

# Visualizations
from matplotlib import pyplot as plt
import seaborn as sb

# Preparation
from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ML Model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix


def visualizations(training_losses, validation_losses, cm, epochs):    
    # Training error per epoch   
    plt.figure(0)
    plt.plot(epochs, training_losses, color='r', linestyle='solid', linewidth=1)      

    plt.title('Error per Epoch (Training)', loc='center', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('error', fontsize=15)
    plt.axis('tight')  

    # Validation error per epoch
    plt.figure(1)
    plt.plot(epochs, validation_losses, color='r', linestyle='solid', linewidth=1)      

    plt.title('Error per Epoch (Validation)', loc='center', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('error', fontsize=15)
    plt.axis('tight')      
    
    # Confusion Matrix   
    plt.figure(2)    
    sb.heatmap(cm, linewidths=1, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=15)
    plt.xlabel('Output Class', fontsize=14)
    plt.ylabel('Target Class', fontsize=14)     
    
    plt.tight_layout()    
    plt.show() 


def feature_selection(input_set):        
    null_variance = []
    pixels = np.shape(input_set)[1]
    for i in range(pixels):        
        temp = input_set[:, i]        
        summation = np.sum(temp, axis=0) 
        if summation == 0:                       
            null_variance.append(i)   
    input_set = np.delete(input_set, null_variance, axis=1)    
    
    return input_set


def main():  
    # Data & Hyperparameters (Optimal) #
    # 1 - Data Splitting
    training_percentage = 0.8
    validation_percentage = 0.3
    shuffle_data = True

    # 2 - Topology
    hidden_layers = (60, )

    # 3 - Functions
    hidden_activation_func = 'logistic'
    output_activation_func = 'softmax'           

    # 4 - Training
    weights_solver = 'sgd'
    lr_type = 'constant'
    lr = 0.15    
    moment = 0.9
    l2_reg = 0.0001
    max_epochs = 50
    early_stp = True
    max_val_fail = 5

    # Preparation #        
    # Loading data
    (training_set, training_targets), (testing_set, testing_targets) = mnist.load_data() 
   
    # Creating the full dataset
    input_set = np.concatenate([training_set, testing_set])  # Creating the full dataset
    input_targets = np.concatenate([training_targets, testing_targets])

    # Turning matrices into vectors
    input_set = input_set.reshape(70000, 784)   

    # Feature selection based on null variance        
    input_set = feature_selection(input_set) 

    # Normalization Min-Max     
    scaler = MinMaxScaler().fit(input_set)
    input_set = scaler.transform(input_set)    

    # Dividing the data    
    training_set, testing_set, training_targets, testing_targets = train_test_split(input_set, input_targets, train_size=training_percentage, random_state=1)
    training_set, validation_set, training_targets, validation_targets = train_test_split(training_set, training_targets, test_size=validation_percentage, random_state=1)           

    # ML Model #
    # Hyperparameters
    mlp = MLPClassifier(        
        shuffle=True,     
        hidden_layer_sizes=hidden_layers,
        activation=hidden_activation_func,        
        solver=weights_solver,        
        learning_rate=lr_type,
        learning_rate_init=lr,        
        momentum=moment,
        alpha=l2_reg,
        max_iter=max_epochs,                  
        random_state=1                
    )    
    mlp.out_activation_ = output_activation_func     

    # Training    
    network_confusion_matrices = []
    training_losses = []
    validation_losses = []   
    
    classes = np.unique(training_targets, axis=0)    
    val_fail_control = 0    
    print(f'Number of Epochs: {max_epochs}')
    for epoch in range(max_epochs):        
        print(f'\tEpoch {epoch}')
        mlp.partial_fit(training_set, training_targets, classes) 

        # Evaluating metrics stored for each "network state" (epoch)                     
        predictions = mlp.predict(testing_set)
        cm = confusion_matrix(testing_targets, predictions)         
        network_confusion_matrices.append(cm) # Confusion Matrices
        
        # Training Losses
        training_output_prob = mlp.predict_proba(training_set)
        training_loss = log_loss(training_targets, training_output_prob, labels=classes) # Log loss is the default performance function used by sklearn MLPClassifier
        training_losses.append(training_loss)
        
        # Validation Losses
        validation_output_prob = mlp.predict_proba(validation_set)
        validation_loss = log_loss(validation_targets, validation_output_prob, labels=classes)        
        validation_losses.append(validation_loss)
        
        # Early Stopping
        if epoch > 0:
            if validation_losses[epoch] > validation_losses[epoch - 1]:
                val_fail_control += 1
            else:
                val_fail_control = 0    
        if val_fail_control == max_val_fail:
            break     
    epochs = [i for i in range(mlp.n_iter_)]    

    # Testing        
    cm = []
    if val_fail_control == max_val_fail: # Getting the best network state               
        cm = network_confusion_matrices[len(epochs) - max_val_fail - 1]
    else:        
        cm = network_confusion_matrices[len(epochs) - 1]       
    
    print(f'Epochs actually executed: {len(epochs)}')        
    print(f'Accuracy: {round((np.trace(cm) / np.sum(cm)) * 100, 2)} %')
    print(f'Confusion Matrix\n{cm}')          

    # Visualizations #           
    visualizations(training_losses, validation_losses, cm, epochs) 

if __name__ == '__main__':
    main()