# MNIST Data Introduction

MNIST stands for Mixed National Institute of Standards and Technology, which has produced a handwritten digits dataset. This is one of the most researched datasets in machine learning, and is used to classify handwritten digits. This dataset is helpful for predictive analytics because of its sheer size, allowing deep learning to work its magic efficiently. This dataset contains 60,000 training images and 10,000 testing images, formatted as 28 x 28 pixel monochrome images.

# Diffeent Approches 
## Multi - Layer Perceptron (With 1 Hidden Layer)
### Steps to Process Data - 

1. Importing Libraries - We have used Keras, Numpy, Matplotlib, Sklear Libraries

1. Data Preprocessing 
   - Normalizing the Data
   - Flattening the Input Data of (28, 28) to (1, 784)
   - Convering the Output Lable (1) which contains values from [0 - 9], transform it to One-Hot Encoder say a lable is 5 its One Hot Vector will looks like [0, 0, 0, 0, 0, 1, 0, 0, 0, 0 , 0]

1. Keras Implementation 
   - Creating Sequential Model with 1 Hidden Layer 
   - Now usign **model.build** we will add Input Layer Dimesion
     - For each Hidden Layer Activation we wil use **sigmoid** function
     - For Output Layer  Activation
       - If we have **Multi-class Classification** we will use **Softmax**
       - IF we have **Binary Classification** we will use **Sigmoid**
   - **model.summary()** will output the model with all Params required to train and summarize the model picture
     - **Parameters Calculation**
       - W1 matrix - Input Shape * Hidden Layer + Bias * Hidden Layer = (784 * 10) + (1 * 10)
       - W2 matrix - Hidden Layer * Output Label + Bias * Output Label = (10 * 10) + (1 * 10)

    - **Compile Model** / Training Model
        - *Loss* - 'categorical_crossentropy'
        - *Optimizer* - 'sgd'
        - *Accuracy* - 'accuracy'

    - Training Model - We will mention how many epochs to use for training

1. Evalating Mode with evaluate method
    - We wii use the Loss and Accuracy over test/ unseen data to predict how our model perfroms over it
    - *Confusion Matrix* - this will have all class matrix say we have 10 labels than matrix will be 10 * 10 and will create how many classes are classifed properly 

1. Saving Model with file_name = *MLP_1HiddenLayer_SGD.keras*

1. Reuse Model - We can reuse model which will stores all layers and corresponding trained weights which can be use for further

