# Palmer Penguins Classification
Use of Multi-Layer Perceptron Classifier for Palmer Penguins Classification of bill length and flipper length data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EskerOn/palmerPenguinsClassification/blob/main/PenguinClassificator.ipynb)

## Introduction
A multilayer perceptron classifier has been widely used in classification problems, due toto its advantages over a simple perceptron. In this worka multilayer perceptron classifier is used to classify 3 species of penguins according to bill length and flipper length. Using [Palmer Penguins](https://github.com/allisonhorst/palmerpenguins) dataset.

## Goal
Build a MLPClassifier for Palmer Penguins Classification according to bill length and flipper length.

## Technologies
For this project Notebook Python was used as well as `scikit-learn` and `pandas` libraries.

## Documentation

### Rtocsv.r
Here the Palmer Penguins dataset was exported to .csv file.
```r
install.packages("palmerpenguins")
library(palmerpenguins)
df <- data.frame(penguins)
write.csv(df,"Path\\penguins.csv", row.names = FALSE)
```
### PenguinClassificator.ipynb

First read the .csv file, create a dataframe filter the relevant attributes and remove the null values with `dropna()`.

```python
#Read data and cleaning
data = pd.read_csv("palmerPenguinsClassification/penguins.csv")
data = data[['species', 'bill_length_mm', 'flipper_length_mm']]
data=data.dropna()
```
Encoding the species label.
```python    
#Label
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])
```
Now is important to scale the data to work in the range needed for the MLPClassifier. And print a preview of the cleaned and scaled data.
```python
#Scale Data
scaler = StandardScaler()
data[['bill_length_mm', 'flipper_length_mm']] = scaler.fit_transform(data[['bill_length_mm', 'flipper_length_mm']])
print("Cleaned and scaled data preview: ")
print(data.head())
```
Define a subset of the dataset to train the MLPClassifier, in this case 80% (0.8 the complement of test_size), and define the arrays of attributes and labels for training data.
```python
#Split Training data
training_set, test_set = train_test_split(data, test_size = 0.2)
X_train = training_set.iloc[:,1:3].values
Y_train = training_set.iloc[:,0].values
```
Build the Classifier with the following parameters: 
* hidden_layer_sizes=(3,5,3)
* activation = 'relu'
* solver='lbfgs'
* max_iter=3000

```python
#Build the Classifier
classifier = MLPClassifier(hidden_layer_sizes=(3,5,3),activation = 'relu',solver='lbfgs', max_iter=3000)
```
Train the classifier with the training arrays.
```python
#Training the model
classifier.fit(X_train, Y_train)
```        

Define the arrays of attributes and labels for the complete dataset.
```python
#Predict the total data
X_test = data.iloc[:,1:3].values
Y_test = data.iloc[:,0].values
```

Predict with the complete dataset.
```python
print("Input data: ")
print(X_test)
Y_pred = classifier.predict(X_test)
```
Compare the predicted data and create a confusion matrix to obtain the accuracy.
```python
print("Predicted data: ")
print(Y_pred)
print("Values data: ")
print(Y_test)
ConfMat = confusion_matrix(Y_pred, Y_test)
print(f"Accuracy of MLPClassifier : {accuracy(ConfMat)}")
```
Accuracy function with confusion matrix.
```python
# Accuracy Function
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements
```  
