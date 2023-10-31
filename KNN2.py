import csv
import math
from sklearn.metrics import f1_score, precision_score , recall_score
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        if is_float(row1[i]) and is_float(row2[i]):
            distance += (float(row1[i]) - float(row2[i])) ** 2
    return math.sqrt(distance)
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    # print(output_values)
    prediction = max(set(output_values), key=output_values.count)
    return prediction
# k-Nearest Neighbors algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return predictions

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Split a dataset into k folds for cross-validation
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = len(dataset_copy) - 1
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Evaluate an algorithm using cross-validation
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = [item for sublist in train_set for item in sublist]
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores
def f1_score_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = [item for sublist in train_set for item in sublist]
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = f1_score(actual, predicted)*100
        scores.append(accuracy)
    return scores
def precision_score_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = [item for sublist in train_set for item in sublist]
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = precision_score(actual, predicted)*100
        scores.append(accuracy)
    return scores
def f1_score_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = [item for sublist in train_set for item in sublist]
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = f1_score(actual, predicted)*100
        scores.append(accuracy)
    return scores
def recall_score_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = [item for sublist in train_set for item in sublist]
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = recall_score(actual, predicted)*100
        scores.append(accuracy)
    return scores
# Read data from the CSV file and convert to floats
data = []

with open('diabetes.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        if all(is_float(x) for x in row):
            data.append([float(x) for x in row])
minmax = dataset_minmax(data)
normalize_dataset(data,minmax)
#
# Example: Evaluate the KNN algorithm using 5-fold cross-validation
scores = evaluate_algorithm(data, k_nearest_neighbors,5, 8)
f1 = f1_score_algorithm(data,k_nearest_neighbors,5,8)
pre = precision_score_algorithm(data,k_nearest_neighbors,5,8)
recall = recall_score_algorithm(data,k_nearest_neighbors,5,8)
# print('Scores: %s' % scores)
print('Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
print('F1 score: %.3f%%' %(sum(f1) / float(len(f1))))
print('Precision score: %.3f%%' %(sum(pre) / float(len(pre))))
print('Recall score: %.3f%%' %(sum(recall) / float(len(recall))))



new_data = [11,143,94,33,146,36.6,0.254,51]
pr =  predict_classification(data ,new_data, 7)
if(pr == 1.0):
    print("Dự đoán bệnh nhân có bị tiểu đường hay không với dữ liệu được truyền vào: Có")
elif(pr == 0.0):
    print("Dự đoán bệnh nhân có bị tiểu đường hay không với dữ liệu được truyền vào: Không")
    