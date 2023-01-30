import sys
import random
import numpy as np
from matplotlib import pyplot as plt


def dividing_into_classes(N, data_set, data_targets):
    class_1 = []
    class_2 = []
    class_3 = []    
    for row in range(N):
        if data_targets[row, 0] == 0 and data_targets[row, 1] == 0:
            class_1.append(data_set[row])
        elif data_targets[row, 0] == 1 and data_targets[row, 1] == 0:
            class_2.append(data_set[row])
        else:
            class_3.append(data_set[row])
    class_1 = np.array(class_1, dtype='float64')
    class_2 = np.array(class_2, dtype='float64')
    class_3 = np.array(class_3, dtype='float64')  

    return class_1, class_2, class_3


def visualizations(N_training, N_testing, training_set, testing_set, training_targets, testing_targets, weights, weights_record, epochs_record):
    # Frontiers: Rows Equations -> w1*x1 + w2*x2 -b = 0, where b = w0*(-1)
    n1_x1 = np.linspace(-5, 5, 1000)
    n1_x2 = (-weights[1][0] * n1_x1 + weights[0][0]) / weights[2][0]
    n2_x1 = np.linspace(-5, 5, 1000)
    n2_x2 = (-weights[1][1] * n1_x1 + weights[0][1]) / weights[2][1]    
    
    # General plotting configurations    
    plt.style.use('dark_background')    
    figure, axis = plt.subplots(nrows=2, ncols=2)

    # Training Set & Decision Boundaries 
    class_1, class_2, class_3 = dividing_into_classes(N_training, training_set, training_targets)    
        
    axis[0, 0].scatter(class_1[:, 0], class_1[:, 1], marker='.', color='c', s=18, label='Class 1')
    axis[0, 0].scatter(class_2[:, 0], class_2[:, 1], marker='.', color='m', s=18, label='Class 2')
    axis[0, 0].scatter(class_3[:, 0], class_3[:, 1], marker='.', color='y', s=18, label='Class 3')
    axis[0, 0].plot(n1_x1, n1_x2, color='b', linestyle='solid', linewidth=1)
    axis[0, 0].plot(n2_x1, n2_x2, color='r', linestyle='solid', linewidth=1)    

    axis[0, 0].set_title('Training Set', loc='center', fontsize=16)
    axis[0, 0].set_xlabel('x1', fontsize=11)
    axis[0, 0].set_ylabel('x2', fontsize=11)
    axis[0, 0].axis('tight')  
    axis[0, 0].legend(loc='lower left', prop={'size': 9})    

    # Testing Set & Decision Boundaries  
    class_1, class_2, class_3 = dividing_into_classes(N_testing, testing_set, testing_targets)         
    
    axis[0, 1].scatter(class_1[:, 0], class_1[:, 1], marker='.', color='c', s=18, label='Class 1')
    axis[0, 1].scatter(class_2[:, 0], class_2[:, 1], marker='.', color='m', s=18, label='Class 2')
    axis[0, 1].scatter(class_3[:, 0], class_3[:, 1], marker='.', color='y', s=18, label='Class 3')
    axis[0, 1].plot(n1_x1, n1_x2, color='b', linestyle='solid', linewidth=1)
    axis[0, 1].plot(n2_x1, n2_x2, color='r', linestyle='solid', linewidth=1)

    axis[0, 1].set_title('Testing Set', loc='center', fontsize=16)
    axis[0, 1].set_xlabel('x1', fontsize=11)
    axis[0, 1].set_ylabel('x2', fontsize=11)
    axis[0, 1].axis('tight')    
    axis[0, 1].legend(loc='lower left', prop={'size': 9})    

    # Record of synaptic weights        
    # Neuron 1
    axis[1, 0].plot(epochs_record, weights_record[:, 0, 0], color='r', linestyle='solid', linewidth=1, label='w00')
    axis[1, 0].plot(epochs_record, weights_record[:, 1, 0], color='g', linestyle='solid', linewidth=1, label='w10')
    axis[1, 0].plot(epochs_record, weights_record[:, 2, 0], color='b', linestyle='solid', linewidth=1, label='w20')    

    axis[1, 0].set_title('Neuron 1', loc='center', fontsize=16)
    axis[1, 0].set_xlabel('Epochs', fontsize=11)
    axis[1, 0].set_ylabel('Weights', fontsize=11)
    axis[1, 0].axis('tight')    
    axis[1, 0].legend(loc='lower left', prop={'size': 9})
    
    # Neuron 2
    axis[1, 1].plot(epochs_record, weights_record[:, 0, 1], color='r', linestyle='solid', linewidth=1, label='w01')
    axis[1, 1].plot(epochs_record, weights_record[:, 1, 1], color='g', linestyle='solid', linewidth=1, label='w11')
    axis[1, 1].plot(epochs_record, weights_record[:, 2, 1], color='b', linestyle='solid', linewidth=1, label='w21')    

    axis[1, 1].set_title('Neuron 2', loc='center', fontsize=16)
    axis[1, 1].set_xlabel('Epochs', fontsize=11)
    axis[1, 1].set_ylabel('Weights', fontsize=11)
    axis[1, 1].axis('tight')    
    axis[1, 1].legend(loc='lower left', prop={'size': 9})           

    plt.tight_layout()        
    plt.show()


def train_test_generator(N, input_data, training_set_percentage, binary_targets):
    indices = [i for i in range(N)]
    shuffled_indices = random.sample(indices, N)        
    train_number_of_indices = int(N * training_set_percentage)
    
    training_set = []
    testing_set = []
    training_targets = []
    testing_targets = []
    for i in range(N):
        index = shuffled_indices[i]        
        if i < train_number_of_indices:
            training_set.append(input_data[index].tolist())
            training_targets.append(binary_targets[index].tolist())
        else:
            testing_set.append(input_data[index].tolist())
            testing_targets.append(binary_targets[index].tolist())
    
    training_set = np.array(training_set, dtype=np.float64)
    testing_set = np.array(testing_set, dtype=np.float64)
    training_targets = np.array(training_targets, dtype=np.int64)
    testing_targets = np.array(testing_targets, dtype=np.int64)   
    
    return training_set, testing_set, training_targets, testing_targets
    

def perceptron_train(N, m, n, training_set, training_targets, max_epochs, learning_rate, training_type, early_stopping):
    # Bias column in the training set    
    bias = np.full((N, 1), -1.) # Column array with N entries -1
    training_set = np.append(bias, training_set, axis=1) # axis=1 -> append a column

    # Setting up the initial weights     
    weights = np.random.randn(m + 1, n) # m + 1(bias). 'randn' uses the "standard normal distribution"   

    # Weights and Epochs arrays for further analysis
    weights_record = np.array([weights], dtype='float64')
    epochs_record = np.array([0], dtype='int64')  

    # Training Epochs
    for epoch in range(max_epochs):               
        # Shuffling the training set/targets
        shuffler = np.random.permutation(N)
        training_set = training_set[shuffler]
        training_targets = training_targets[shuffler]        
        
        # Computing H = X.W
        output = training_set @ weights # Dimensions -> training_set: N x m + 1, weights: m + 1 x n, H: output: N x n        

        # Computing O = f(H)
        for row in range(N): # f(H) Applying the step function
            for column in range(n):
                if output[row][column] > 0.:
                    output[row][column] = 1
                else:
                    output[row][column] = 0        
        output = output.astype('int64') # Casting output to int64
        
        # Computing Error = O - T
        error = output - training_targets

        # Weights update
        if training_type == 0: # Batch
            training_set_transposed = np.transpose(training_set)
            weights = weights - learning_rate * (training_set_transposed @ error)/N

        else: # Sequential
            for row in range(m + 1):
                for column in range(n):                    
                    weights[row][column] = weights[row][column] - learning_rate * (training_set[:, row] @ error[:, column])
        
        # Updating Records
        weights_record = np.append(weights_record, [weights], axis=0)
        epochs_record = np.append(epochs_record, [epoch + 1], axis=0)   

        # Early Stopping
        hit = 0
        miss = 0
        for row in range(N):            
            flag = 0
            for column in range(n):
                if error[row][column] == 0:
                    flag += 1
            if flag == n:
                hit += 1
            else:
                miss += 1        
        
        hit_percentage = (hit / N) * 100
        if hit_percentage >= early_stopping:              
            break       

    print(f'\nFinal Weights:\n{weights}')

    return weights, weights_record, epochs_record


def perceptron_test(N, n, testing_set, testing_targets, weights):
        # Bias column in the testing set    
        bias = np.full((N, 1), -1.) # Column array with N entries -1
        testing_set = np.append(bias, testing_set, axis=1) # axis=1 -> append a column

        # Computing H = X.W
        output = testing_set @ weights # Dimensions -> testing_set: N x m + 1, weights: m + 1 x n, H: output: N x n        

        # Computing O = f(H)
        for row in range(N): # O(H) Applying the step function
            for column in range(n):
                if output[row][column] > 0.:
                    output[row][column] = 1
                else:
                    output[row][column] = 0         
        output = output.astype('int64') # Casting output to int64

        error = output - testing_targets        

        # Evalutation
        hit = 0
        miss = 0
        for row in range(N):            
            flag = 0
            for column in range(n):
                if error[row][column] == 0:
                    flag += 1
            if flag == n:
                hit += 1
            else:
                miss += 1        
        accuracy = (hit / N) * 100

        print(f'\nAccuracy: {round(accuracy, 2)}%')


def main():
    # Initial configuration #  
    learning_rate = float(input('Learning Rate: '))    
    if learning_rate < 0. or learning_rate > 1.:
        print('Invalid learning rate.')
        sys.exit()

    max_epochs = int(input('Max Epochs: '))      
    if max_epochs < 0.:
        print('Invalid max epochs.')    
    
    training_set_percentage = float(input('Training Set(%): ')) / 100       
    if training_set_percentage < 0. or training_set_percentage > 100.:
        print('Invalid training set.')
        sys.exit()   

    early_stopping = float(input('Acceptable HIT percentage (training): '))       
    if early_stopping < 0. or early_stopping > 100.:
        print('Invalid acceptable hit percentage.')
        sys.exit() 

    training_type = 0        

    # Input data and targets #
    input_data = np.genfromtxt('DATA.txt', delimiter=',', dtype=np.float64)
    targets = np.genfromtxt('TARGETS.txt', delimiter=',', dtype=np.int64)    
    
    N, m = input_data.shape # 'N' instances of the input and 'm' variables
    targets_rows = targets.size     

    # Definning the targets in a binary format #
    binary_targets = []
    
    for row in range(targets_rows): # Encoding the classes: 1 -> 00, 2 -> 10, 3 -> 11
        binary_targets.append([])
        if targets[row] == 1:
            binary_targets[row].append(0)
            binary_targets[row].append(0)
        elif targets[row] == 2:
            binary_targets[row].append(1)
            binary_targets[row].append(0)
        else:
            binary_targets[row].append(1)
            binary_targets[row].append(1)    

    binary_targets = np.array(binary_targets, dtype=np.int64)
    N, n = binary_targets.shape # N instances of the input and 'n' outputs that composes the target -> 'n' neurons required        

    # Creating the training and testing sets #   
    training_set, testing_set, training_targets, testing_targets = train_test_generator(N, input_data, training_set_percentage, binary_targets)   

    # Perceptron Training #
    weights, weights_record, epochs_record = perceptron_train(int(N * training_set_percentage), m, n, training_set, training_targets, max_epochs, learning_rate, training_type, early_stopping) # The N passed is a percentage (for training) of the original N    

    # Perceptron Testing #
    perceptron_test((N - int(N * training_set_percentage)), n, testing_set, testing_targets, weights)      
    
    # Visualizations
    visualizations(int(N * training_set_percentage), (N - int(N * training_set_percentage)), training_set, testing_set, training_targets, testing_targets, weights, weights_record, epochs_record)


if __name__ == "__main__":
    main()
