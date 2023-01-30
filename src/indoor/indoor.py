import numpy as np
import pandas as pd
import random

# Visualizations #
from matplotlib import pyplot as plt
import seaborn as sns
from pretty_confusion_matrix import pp_matrix

# Preparation #
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical

# ML Model #
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.naive_bayes import CategoricalNB

# Model Evaluation #
from sklearn.metrics import confusion_matrix


def terminal_message(msg):
    row = len(msg)
    design = ''.join(['+'] + ['-' *row] + ['+'])
    result = design + '\n'"|"+msg+"|"'\n' + design
    return result


def main():
    # Seeds #
    random.seed(10)
    np.random.seed(10)

    # User Input #   
    print('\n' + terminal_message('BLUE CONFUSION MATRIX -> NAIVE BAYES'))
    print(terminal_message('BROWN CONFUSION MATRIX -> MLP'))
    with_pca = input('Perform Naive Bayes and MLP (choose 1 or 2):\n\t1 - WITH PCA\n\t2 - WITHOUT PCA\n> ')

    if with_pca != '1' and with_pca != '2':
        print('Invalid choice.')
    
    if with_pca == '1':
        with_pca = True
    else:
        with_pca = False
    
    # Optimal Configuration #
    # Naive Bayes
    training_percentage_nb = 0.8

    # MLP
    training_percentage_mlp = 0.8
    validation_percentage_mlp = 0.2
    
    hidden_neurons = 100
    output_neurons = 7
    lr = 0.025 
    hidden_activation_function = 'tanh'
    output_activation_function = 'softmax'
    performance_function = 'categorical_crossentropy'
    metrics_ = 'accuracy'
    opt = Adam(learning_rate=lr)    
    max_epochs = 20
    max_val_fail = 5

    # Loading Data #
    input_data = np.genfromtxt('wifi_scans.csv', delimiter=',', dtype=np.float64)

    targets = input_data[1:, 137]  
    input_data = np.delete(input_data, [0], axis=0) # Removing the indices row   
    input_data = np.delete(input_data, [136, 137], axis=1) # Removing the FLOOR(targets) and POINTS column   

    # Features Selection #
    selector = VarianceThreshold() # Removing all constant features
    input_data = selector.fit_transform(input_data)   

    # Correlation between features #
    df = pd.DataFrame(data=input_data)
    df = df.drop([i for i in range(25, input_data[0].size)], axis=1) # Utilizing only the first 25 features

    ax = plt.axes()
    sns.heatmap(df.corr(), cmap='Blues', ax=ax)
    ax.set_title('Features Correlation (first 25)')   
  
    # Splitting data #
    training_set_nb, testing_set_nb, training_targets_nb, testing_targets_nb = train_test_split(input_data, targets, train_size=training_percentage_nb, shuffle=True, random_state=1)
    training_set_mlp, testing_set_mlp, training_targets_mlp, testing_targets_mlp = train_test_split(input_data, targets, train_size=training_percentage_mlp, shuffle=True, random_state=1)    
    
    # PCA #
    pca = PCA(random_state=1) 
    if with_pca:  
        # Standardization -> sd = 1 and mean = 0
        # scaler = StandardScaler()
        # training_set_nb = scaler.fit_transform(training_set_nb)
        # testing_set_nb = scaler.transform(testing_set_nb)
        # training_set_mlp = scaler.transform(training_set_mlp)
        # testing_set_mlp = scaler.transform(testing_set_mlp)

        # Performing PCA 
        training_set_nb = pca.fit_transform(training_set_nb)
        testing_set_nb = pca.transform(testing_set_nb)
        training_set_mlp = pca.transform(training_set_mlp)
        testing_set_mlp = pca.transform(testing_set_mlp)

        # Removing the PCs with small variances
        num_features_rm = 50
        pc_variances = pca.explained_variance_ratio_
        sorted_variances = np.argsort(pc_variances)
        low_var_features = [sorted_variances[i] for i in range(0, num_features_rm)]
        training_set_nb = np.delete(training_set_nb, low_var_features, axis=1)   
        testing_set_nb = np.delete(testing_set_nb, low_var_features, axis=1)    
        training_set_mlp = np.delete(training_set_mlp, low_var_features, axis=1) 
        testing_set_mlp = np.delete(testing_set_mlp, low_var_features, axis=1) 

    # Discretization, Continuous data -> Categorical data - Naive Bayes #    
    kbins = KBinsDiscretizer(n_bins=120, encode='ordinal', strategy='uniform', random_state=1)
    training_set_nb = kbins.fit_transform(training_set_nb)
    testing_set_nb = kbins.transform(testing_set_nb)    

    # Normalization and One-Hot encoding - MLP 
    scaler = MinMaxScaler(feature_range=(0, 1)) # Normalizing the input (set)
    training_set_mlp = scaler.fit_transform(training_set_mlp)
    testing_set_mlp = scaler.transform(testing_set_mlp)

    training_targets_mlp = to_categorical(training_targets_mlp)
    testing_targets_mlp = to_categorical(testing_targets_mlp) 

    # Naive Bayes Model #   
    nb_model = CategoricalNB()
    nb_model.fit(training_set_nb, training_targets_nb)
    predictions_nb = nb_model.predict(testing_set_nb)
    accuracy = nb_model.score(testing_set_nb, testing_targets_nb)       

    # MLP Model #    
    model = Sequential()   

    model.add(
        Dense(
            units=hidden_neurons,                      
            activation=hidden_activation_function,
            input_shape=(training_set_mlp.shape[1], )
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
        loss=performance_function,        
        metrics=metrics_          
    )    
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=max_val_fail, verbose=0)
    history = model.fit(
        x=training_set_mlp,
        y=training_targets_mlp,
        validation_split=validation_percentage_mlp,        
        epochs=max_epochs,
        shuffle=True,
        verbose=1,
        callbacks=[early_stopping]      
    )   

    loss, acc = model.evaluate(testing_set_mlp, testing_targets_mlp)
    predictions_mlp = model.predict(x=testing_set_mlp, verbose=0)

    # Converting from one-hot to integers (floors: 4, 5, 6)
    output_targets = []
    true_targets = []
    for i in range(testing_targets_mlp.shape[0]):
        output_targets.append(np.argmax(testing_targets_mlp[i]))
        true_targets.append(np.argmax(predictions_mlp[i]))    

    # Naive Bayes PCs
    if with_pca:
        pc_variances = pca.explained_variance_ratio_
        pc_variables = [i for i in range(pc_variances.size)]
        
        plt.figure(2)
        plt.plot(pc_variables, pc_variances, color='r', linestyle='solid', linewidth=1)      

        plt.title('% Total Variance x PC - Naive Bayes', loc='center', fontsize=15)
        plt.xlabel('PC', fontsize=15)
        plt.ylabel('% Total Variance', fontsize=15)
        plt.axis('tight')

        pc_variance_sum = [0 for i in range(pc_variances.size)]
        for i in range(pc_variances.size):
            for j in range(0, i):
                pc_variance_sum[i] += pc_variances[j] 

        plt.figure(3)
        plt.plot(pc_variables, pc_variance_sum, color='r', linestyle='solid', linewidth=1)      

        plt.title('% Total Variance x Number of PCs - Naive Bayes', loc='center', fontsize=15)
        plt.xlabel('Number of PCs', fontsize=15)
        plt.ylabel('% Total Variance', fontsize=15)
        plt.axis('tight')       

    # Training and Validation error (Cross Entropy) x Epoch
    training_error = history.history['loss']
    validation_error = history.history['val_loss']  
    epochs = [i for i in range(len(training_error))] 
    plt.figure(4)     

    plt.plot(epochs, training_error, color='r', linestyle='solid', linewidth=1, label='training_error')      
    plt.plot(epochs, validation_error, color='black', linestyle='solid', linewidth=1, label='validation_error')      

    plt.title('Error (Cross Entropy) per Epoch - MLP', loc='center', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('error', fontsize=15)
    plt.axis('tight')
    plt.legend(loc='upper left', prop={'size': 9})

    # Naive Bayes Confusion Matrix
    cm_nb = confusion_matrix(testing_targets_nb, predictions_nb) 
    df_cm = pd.DataFrame(cm_nb, index=range(4, 7), columns=range(4, 7))
    pp_matrix(df_cm, cmap='Blues')  
   
    # MLP Confusion Matrix
    cm_mlp = confusion_matrix(true_targets, output_targets) 
    df_cm = pd.DataFrame(cm_mlp, index=range(4, 7), columns=range(4, 7))
    pp_matrix(df_cm)  


if __name__ == '__main__':
    main()