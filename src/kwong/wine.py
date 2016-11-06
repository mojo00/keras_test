# Multiclass Classification with the Iris Flowers Dataset
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("wine.txt", index_col = False)
col_names = list(dataframe)
dataset = dataframe.values
X = dataset[:,1:].astype(float)
Y = dataset[:,0]

# data = np.genfromtxt('wine.txt', delimiter = ',', names = True)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# define baseline model

input_dimension = X.shape[1]
output_dimension = dummy_y.shape[1]

hidden_layer_dimension = int(np.mean([input_dimension, output_dimension]))

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(hidden_layer_dimension, input_dim=input_dimension, init='normal', activation='relu'))
	model.add(Dense(output_dimension, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
kfold = KFold(n=len(X), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
