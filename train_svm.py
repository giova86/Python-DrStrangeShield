from sklearn import svm
import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-i", "--file", dest="filename", default='data/hand_position.csv',
                    help="PATH of training FILE")
parser.add_argument("-t", "--test", dest="test_size", default=0.3, type=float,
                    help="Test size. A number between 0 and 1. default value is 0.3")
parser.add_argument("-o", "--output", dest="output_file", default='model_svm',
                    help="Name of the saved model. default is 'model_svm'. the model will be saved in the 'models' folder with .sav extension")
args = parser.parse_args()

print(f'--> Loading dataset from {args.filename}')
df = pd.read_csv(f'{args.filename}', index_col=0)
print('DONE')

# prepare X and y variables
X = np.array(df.iloc[:,:-1])
y = np.array(df['y'])
print('')
print('--> Summary')
print(f'Number of samples: {len(X)}')
print(f'Number of features: {X.shape[1]}')
print(f'Number of classes: {len(np.unique(y))}')

# train test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size) # 70% training and 30% test
print(f'Number of samples for Training: {len(X_train)}')
print(f'Number of samples for Test: {len(X_test)}')

# define and train SVM model
clf = svm.SVC(probability=True)
print('')
print('--> Training SVM Model')
clf.fit(X_train, y_train)
print('DONE')

#Predict the response for train & test datasets
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print('')
print('--> Model performances')
print("Accuracy Train:",metrics.accuracy_score(y_train, y_train_pred))
print("Accuracy Test: ",metrics.accuracy_score(y_test, y_test_pred))

# save the model to disk
print('')
print(f'--> Saving model in "models/{args.output_file}.sav"')
filename = f'models/{args.output_file}.sav'
pickle.dump(clf, open(filename, 'wb'))
print("DONE")
print('')
