import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define the parameter grid to search through
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'auto'],
    'kernel': ['linear', 'rbf', 'poly']
}

# Create an SVM classifier
svm_model = SVC()

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train a new SVM model with the best parameters
best_model = SVC(**best_params)
best_model.fit(x_train, y_train)

# Evaluate the model on the test set
y_predict = best_model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the best SVM model 
f = open('model.p', 'wb')
pickle.dump({'model': best_model}, f)
f.close()






