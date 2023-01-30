import random
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf 
from sklearn.metrics import confusion_matrix


def visualizations(max_epochs, error_per_epoch_0, n_0, output_0, error_per_epoch_1, n_1, output_1, testing_targets):   
    # One-hot Enc. Error x Epoch
    plt.figure(0)
    epochs = [i for i in range(max_epochs)]
    plt.plot(epochs, error_per_epoch_0, color='r', linestyle='solid', linewidth=1)      

    plt.title('Error per Epoch (One-hot Enc.)', loc='center', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('error', fontsize=15)
    plt.axis('tight')               
    
    # One-hot Enc. Confusion Matrix
    plt.figure(1)
    l = [i for i in range(10)]
    out_0 = []
    for i in range(10000):
        for j in range(n_0):
            if output_0[i][j] == 1:
                out_0.append(j)
                break    
    
    cm = confusion_matrix(testing_targets, out_0, labels=l)        
    cm = np.transpose(cm)        
    print(f'\n- Confusion Matrix (One-hot Enc.)\n{cm}')        
    
    sns.heatmap(cm, linewidths=1, annot=True, fmt='g')
    plt.title('Confusion Matrix (One-hot Enc.)', fontsize=15)
    plt.xlabel('Target Class', fontsize=14)
    plt.ylabel('Output Class', fontsize=14)   

    # Binary Enc. Error x Epoch
    plt.figure(2)
    epochs = [i for i in range(max_epochs)]
    plt.plot(epochs, error_per_epoch_1, color='r', linestyle='solid', linewidth=1)      

    plt.title('Error per Epoch (Binary Enc.)', loc='center', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('error', fontsize=15)
    plt.axis('tight')               
    
    # Binary Enc. Confusion Matrix
    plt.figure(3)
    l = [i for i in range(10)]
    out_1 = []
    for i in range(10000):
        summation = 0
        for j in range(n_1):            
            summation += output_1[i][j] * 2**(n_1 -1 -j)
        out_1.append(summation)                   
    
    cm = confusion_matrix(testing_targets, out_1, labels=l)        
    cm = np.transpose(cm)
    print(f'\n- Confusion Matrix (Binary Enc.)\n{cm}')     
    
    sns.heatmap(cm, linewidths=1, annot=True, fmt='g')
    plt.title('Confusion Matrix (Binary Enc.)', fontsize=15)
    plt.xlabel('Target Class', fontsize=14)
    plt.ylabel('Output Class', fontsize=14)    
    
    plt.tight_layout()    
    plt.show()  


def perceptron_train(N, m, n, training_set, training_targets, testing_set, testing_targets, max_epochs, learning_rate, training_type):
    # Bias column in the training set    
    bias = np.full((N, 1), -1.) # Column array with N entries -1
    training_set = np.append(bias, training_set, axis=1) # axis=1 -> append a column
   
    # Setting up the initial weights     
    weights = np.random.randn(m + 1, n) # m + 1(bias). 'randn' uses the "standard normal distribution"    
    
    # Training Epochs
    error_per_epoch = []
    for epoch in range(max_epochs):               
        print(f'\tEpoch {epoch}')            
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
                elif output[row][column] < 0.:
                    output[row][column] = 0   
        output = output.astype('int64') 
       
        # Computing Error = O - T
        error = output - training_targets

        # Weights update
        if training_type == 0: # Batch
            training_set_transposed = np.transpose(training_set)
            weights = weights - learning_rate * ((training_set_transposed @ error) / N)

        else: # Sequential
            for i in range(m + 1):
                for j in range(n):                    
                    weights[i][j] = weights[i][j] - learning_rate * (training_set[:, i] @ error[:, j])             
        
        # Calculating the error per epoch
        output, accuracy = perceptron_test(10000, n, testing_set, testing_targets, weights)
        miss_percentage = (100 - accuracy)
        miss_amount = (miss_percentage/100) * 10000
        error_per_epoch.append(miss_amount)              

    return weights, error_per_epoch


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
                elif output[row][column] < 0.:
                    output[row][column] = 0         
        output = output.astype('int64') # Casting output to int64       
        
        if n == 10:
            for i in range(N):
                flag = 0        
                for j in range(n): # More than one 1, choosing the first one
                    if output[i][j] == 1 and flag == 0:
                        flag = 1
                    elif output[i][j] == 1 and flag == 1:
                        output[i][j] = 0       
                if flag == 0: # Full with only zeros, choosing randomly
                    rand_num = random.randrange(10)
                    output[i][rand_num] = 1
        elif n == 4:            
            for i in range(N):
                summation = 0
                for j in range(n):
                    summation += output[i][j] * 2**(n-1-j)                        
                if summation > 9: # Choosing randomly 
                    rand_num = random.randrange(10)                    
                    binary_number = [int(i) for i in list('{0:04b}'.format(rand_num))]  
                    binary_number = np.array(binary_number, dtype='int64')             
                    output[i] = binary_number        

        # Computing Error = O - T
        error = output - testing_targets  

        # Evalutation
        hit = 0        
        for row in range(10000):            
            flag = 0
            for column in range(n):
                if error[row][column] == 0:
                    flag += 1
            if flag == n:
                hit += 1            
        accuracy = (hit / 10000) * 100       

        return output, accuracy


def targets_encoding(targets, type):
    targets_rows = targets.size    
    binary_targets = []    
    
    if type == 0:        
        for row in range(targets_rows): 
            temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            temp[targets[row]] = 1 
            binary_targets.append(temp)                
    
    elif type == 1: 
        for row in range(targets_rows):
            binary_number = [int(i) for i in list('{0:04b}'.format(targets[row]))]
            binary_targets.append(binary_number)     

    binary_targets = np.array(binary_targets, dtype='int64')    
    return binary_targets


def normalization(N_1, N_2, m, training_set, testing_set):
    minimum = min(training_set.min(), testing_set.min())
    maximum = max(training_set.max(), testing_set.max())

    MIN = np.full((N_1, m), minimum, dtype='float64')
    MAX_MIN = np.full((N_1, m), 1 / (maximum - minimum), dtype='float64')
    training_set_norm = (training_set - MIN) * MAX_MIN

    MIN = np.full((N_2, m), minimum, dtype='float64')
    MAX_MIN = np.full((N_2, m), 1 / (maximum - minimum), dtype='float64')
    testing_set_norm = (testing_set - MIN) * MAX_MIN

    return training_set_norm, testing_set_norm


def feature_selection(m, training_set, testing_set):
    counter = 0
    null_variance = []
    for i in range(m):        
        temp = training_set[:, i]
        temp_2 = testing_set[:, i]
        summation = np.sum(temp, axis=0) + np.sum(temp_2, axis=0)
        if summation == 0:
            counter += 1            
            null_variance.append(i)   
    training_set = np.delete(training_set, null_variance, axis=1)
    testing_set = np.delete(testing_set, null_variance, axis=1)
    
    return (m - counter), training_set, testing_set


def main():    
    # Initial configuration #  
    learning_rate = 0.25     
    max_epochs = 19
    training_type = 1

    # Training & Testing Sets/Targets #    
    (training_set, training_targets), (testing_set, testing_targets) = tf.keras.datasets.mnist.load_data() # Total instances 70000(100%), approximately: 60000(85%) -> Training , 10000(15%) -> Testing    
    N = 60000
    m = 784           
    
    training_set_adj = training_set.reshape(60000, 784) # Adjusting matrices to vectors    
    testing_set_adj = testing_set.reshape(10000, 784) # Adjusting matrices to vectors    
    m, training_set_fs, testing_set_fs = feature_selection(m, training_set_adj, testing_set_adj) # Feature Selection based on null variance        
    training_set_norm, testing_set_norm = normalization(60000, 10000, m, training_set_fs, testing_set_fs) # Normalizing     
    
    # Definning the targets in a binary format #  
    # One-hot Encoding  
    training_targets_0 = targets_encoding(training_targets, 0)
    testing_targets_0 = targets_encoding(testing_targets, 0)    
    n_0 = training_targets_0[0].size 
    
    # Binary Encoding
    training_targets_1 = targets_encoding(training_targets, 1)
    testing_targets_1 = targets_encoding(testing_targets, 1)            
    n_1 = training_targets_1[0].size 

    # Perceptron Training #
    print('\n- Training Epochs (One-hot Enc.)')
    weights_0, error_per_epoch_0 = perceptron_train(60000, m, n_0, training_set_norm, training_targets_0, testing_set_norm, testing_targets_0, max_epochs, learning_rate, training_type)    

    print('\n- Training Epochs (Binary Enc.)')
    weights_1, error_per_epoch_1 = perceptron_train(60000, m, n_1, training_set_norm, training_targets_1, testing_set_norm, testing_targets_1, max_epochs, learning_rate, training_type)    
    
    # Perceptron Testing #
    print('\nOne-hot Enc. -> ', end='')
    output_0, accuracy_0 = perceptron_test(10000, n_0, testing_set_norm, testing_targets_0, weights_0)   
    print(f'Accuracy: {round(accuracy_0, 2)}%')
    
    print('Binary Enc. -> ', end='')
    output_1, accuracy_1 = perceptron_test(10000, n_1, testing_set_norm, testing_targets_1, weights_1)
    print(f'Accuracy: {round(accuracy_1, 2)}%')

    # Visualizations # 
    visualizations(max_epochs, error_per_epoch_0, n_0, output_0, error_per_epoch_1, n_1, output_1, testing_targets)


if __name__=="__main__":
    main()

