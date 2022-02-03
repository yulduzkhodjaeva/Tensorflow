import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

dataframe = pd.read_csv("UNB_GT_IDS_12000_14.csv", header=0)
dataset = dataframe.values
X = dataset[:, 0:13].astype(float)
Y = dataset[:, 13].astype(float)


def create_baseline():
    model = Sequential()
    model.add(Dense(60, activation='relu', input_dim=13))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=1)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, Y, cv=kfold, verbose=1)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Accuracy of 99.48%





