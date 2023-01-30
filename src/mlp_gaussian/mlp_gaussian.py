import sys
import random
import numpy as np

# Visualizations
from matplotlib import pyplot as plt
import seaborn as sb

# Preparation
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
    sb.heatmap(cm, linewidths=1, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.title('Confusion Matrix', fontsize=15)
    plt.xlabel('Output Class', fontsize=14)
    plt.ylabel('Target Class', fontsize=14)     
    
    plt.tight_layout()    
    plt.show() 


def hid_activation_function(func_id):
    if func_id == 1:
        return 'identity'
    elif func_id == 2:
        return 'logistic'
    elif func_id == 3:
        return 'tanh'
    elif func_id == 4:
        return 'relu'
    else:
        return -1        


def out_activation_function(func_id):
    if func_id == 1:
        return 'softmax'
    elif func_id == 2:
        return 'identity'
    elif func_id == 3:
        return 'logistic'        
    else:
        return -1 


def user_config(training_percentage, validation_percentage, lr, max_epochs, max_val_fail, hidden_neurons, hidden_activation_func, output_activation_func):    
    print('Training percentage:')
    training_percentage = float(input('> ')) / 100       
    if training_percentage < 0. or training_percentage > 1.:
        print('Invalid training percentage.')
        sys.exit()   
    
    print('Validation percentage:')    
    validation_percentage = float(input('> ')) / 100       
    if validation_percentage < 0. or validation_percentage > 1.:
        print('Invalid validation percentage.')
        sys.exit()       
    
    print('Learning rate:')
    lr = float(input('> '))    
    if lr < 0. or lr > 1.:
        print('Invalid learning rate.')
        sys.exit()
    
    print('Max epochs:')
    max_epochs = int(input('> '))    
    if max_epochs < 0:
        print('Invalid max epochs.')
        sys.exit()

    print('Early Stopping (maxValFail):')
    max_val_fail = int(input('> ')) 
    if max_val_fail < 0:
        print('Invalid maxValFail.')
        sys.exit()

    print('Number of hidden neurons:')
    hidden_neurons = int(input('> '))    
    if hidden_neurons < 0:
        print('Invalid number of hidden neurons.')
        sys.exit()

    print(f'Hidden activation function (choose a number):\n\t1 - Identity (linear)\n\t2 - Logistic\n\t3 - Hyperbolic Tangent\n\t4 - Rectified Linear Unit')    
    hidden_activation_func = int(input('> '))
    hidden_activation_func = hid_activation_function(hidden_activation_func)
    
    if hidden_activation_func == -1:
        print('Invalid hidden activation function.')
        sys.exit()

    print(f'Output activation function (choose a number):\n\t1 - Softmax\n\t2 - Identity (linear)\n\t3 - Logistic')    
    output_activation_func = int(input('> '))
    output_activation_func = out_activation_function(output_activation_func)
    
    if output_activation_func == -1:
        print('Invalid output activation function.')
        sys.exit()  
    
    return training_percentage, validation_percentage, lr, max_epochs, max_val_fail, hidden_neurons, hidden_activation_func, output_activation_func


def optimal_config(training_percentage, validation_percentage, lr, max_epochs, max_val_fail, hidden_neurons, hidden_activation_func, output_activation_func):
    training_percentage = 0.76
    validation_percentage = 0.3
    lr = 0.15
    max_epochs = 200
    max_val_fail = 10
    hidden_neurons = 9
    hidden_activation_func = 'logistic'
    output_activation_func = 'softmax'

    return training_percentage, validation_percentage, lr, max_epochs, max_val_fail, hidden_neurons, hidden_activation_func, output_activation_func


def main():
    # Initial configurations #
    print('\nConfiguration option:\n\t1 - User configuration (manually topology/hyperparameters setting up)\n\t2 - Optimal configuration')
    config_option = int(input('> '))
    if config_option != 1 and config_option != 2:
        print('Invalid configuration option.')
        sys.exit()

    training_percentage, validation_percentage, lr, max_epochs, max_val_fail, hidden_neurons, hidden_activation_func, output_activation_func = '', '', '', '', '', '', '', ''    

    if config_option == 1:
        training_percentage, validation_percentage, lr, max_epochs, max_val_fail, hidden_neurons, hidden_activation_func, output_activation_func = user_config(training_percentage, validation_percentage, lr, max_epochs, max_val_fail, hidden_neurons, hidden_activation_func, output_activation_func)

    else:
        training_percentage, validation_percentage, lr, max_epochs, max_val_fail, hidden_neurons, hidden_activation_func, output_activation_func = optimal_config(training_percentage, validation_percentage, lr, max_epochs, max_val_fail, hidden_neurons, hidden_activation_func, output_activation_func)

    print('\n--------------- Results ---------------')
    print('Topology/Hyperparameters:')
    print(f'\ttraining_percent={training_percentage}\n\tvalidation_percentage={validation_percentage}\n\tlearning_rate={lr}\n\tmax_epochs={max_epochs}\n\tmax_val_fail={max_val_fail}\n\thidden_neurons={hidden_neurons}\n\thidden_activation_func={hidden_activation_func}\n\toutput_activation_func={output_activation_func}\n')
    
    # Preparation #
    # Loading data 
    input_data = np.genfromtxt('DATA.txt', delimiter=',', dtype=np.float64)
    targets = np.genfromtxt('TARGETS.txt', delimiter=',', dtype=np.int64)    
    
    # Dividing the data 
    training_set, testing_set, training_targets, testing_targets = train_test_split(input_data, targets, train_size=training_percentage, random_state=1)
    training_set, validation_set, training_targets, validation_targets = train_test_split(training_set, training_targets, test_size=validation_percentage, random_state=1)    

    # ML Model #
    # Topology and Hyperparameters
    mlp = MLPClassifier(        
        hidden_layer_sizes=(hidden_neurons, ),
        learning_rate='constant',
        learning_rate_init=lr,
        shuffle=True,        
        solver='sgd',
        activation=hidden_activation_func,        
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
    for epoch in range(max_epochs):
        mlp.partial_fit(training_set, training_targets, classes) 

        # Evaluating metrics stored for each "network state" (epoch)                     
        predictions = mlp.predict(testing_set)
        cm = confusion_matrix(testing_targets.argmax(axis=1), predictions.argmax(axis=1))         
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
    print(f'Confusion Matrix\n{cm}')    

    # Visualizations #           
    visualizations(training_losses, validation_losses, cm, epochs)  


if __name__ == "__main__":
    main()
