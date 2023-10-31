from random import randrange
from csv import reader
from math import exp
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
def load_csv(data):
	dataset = list()
	with open(data, 'r') as file:
		csv_reader = reader(file)
		next(csv_reader)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def convert_str_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

def normalize(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)

	return dataset_split

def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def training(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	f1s = list()
	recalls = list()
	precisions = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		# accuracy = accuracy_metric(actual, predicted)
		accuracy = accuracy_score(actual, predicted)
		precision = precision_score(actual, predicted)
		recall = recall_score(actual, predicted)
		f1 = f1_score(actual, predicted)
		scores.append(accuracy)
		precisions.append(precision)
		recalls.append(recall)
		f1s.append(f1)
	return scores, precisions, recalls, f1s

def predict(row, weights):
	z = weights[0]
	for i in range(len(row)-1):
		z += weights[i + 1] * row[i]
	return 1.0 / (1.0 + exp(-z))

def update_weights(train, lr, iter):
	weights = [0.0 for i in range(len(train[0]))]
	for i in range(iter):
		for row in train:
			z = predict(row, weights)
			error = row[-1] - z
			weights[0] = weights[0] + lr * error * z * (1.0 - z)
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + lr * error * z * (1.0 - z) * row[i]
	return weights

def logistic_regression(train, test, lr, iter):
	predictions = list()
	weights = update_weights(train, lr, iter)
	for row in test:
		z = predict(row, weights)
		z = round(z)
		predictions.append(z)
	return (predictions)



data= 'diabetes.csv'
dataset = load_csv(data)

for i in range(len(dataset[0])):
	convert_str_to_float(dataset, i)

minmax = minmax(dataset)
normalize(dataset, minmax)
#
n_folds = 10
lr = 0.1
iter = 100

scores,precisions, recalls, f1s  = training(dataset, logistic_regression, n_folds, lr, iter)

print('Scores: %s' % scores)
print('Tỉ lệ dự đoán đúng của logistic:')
print('Accuracy: %.3f%%' % ((sum(scores)/n_folds) * 100.0))
print('Precision: %.3f%%' % ((sum(precisions)/n_folds) * 100.0))
print('Recall: %.3f%%' % ((sum(recalls)/n_folds) * 100.0 ))
print('f1_scores: %.3f%%' % ((sum(f1s)/n_folds) * 100.0 ))
