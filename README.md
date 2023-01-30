## Supervised Learning with Python
### What does it do?
Implementation in Python of data analysis and machine learning programs utilizing a supervised learning approach over different datasets. The focus was on Perceptron, Multi-Layer Perceptron, and Naive Bayes algorithms.


### How to Use?
#### Dependencies
```bash
* keras==2.10.0
* matplotlib==3.6.2
* numpy==1.23.4
* pandas==1.5.1
* pretty_confusion_matrix==0.1.1
* scikit_learn==1.2.1
* seaborn==0.11.2
* tensorflow==2.10.0
```
- To install the dependencies, use the **requirements.txt** file present in the project folder.
    - Within the **project** folder, Run:
        ```bash
        $ pip install -r requirements.txt
        ``` 

#### Implementations
- Six projects were made involving the already mentioned algorithms. Each directory represents what follows:
    - **src/percepetron_gaussian**
        - Implementation of the Perceptron algorithm and classification of Gaussian distributions.

    - **src/percepetron_ocr**
        - Optical character recognition with the Perceptron algorithm.

    - **src/mlp_gaussian**
        - Multi-Layer Perceptron algorithm for classification of Gaussian distributions.

    - **src/mlp_ocr**
        - Optical character recognition with the Multi-Layer Perceptron algorithm. 


    - **src/mlp_time_series**
        - MLP to predict daily values of COVID deaths in Brazil, using previous K values (days).
        - Training Period: January to April 2022
        - Testing Period: May 2022
        
    - **src/indoor**
        - Comparison of MLP and Naive Bayes algorithms when classifying the floor in an indoor positional system.


