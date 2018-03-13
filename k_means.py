import csv
import numpy as np

def load_data(inputFile):
	reader = csv.reader(open(inputFile))
	df = np.array([row for row in reader if row])
	return df[:,:-1],df[:,-1]

def k_means(k, X):
	pass

def main():
	inputFile = 'iris.data.txt'
	X, y = load_data(inputFile)
	


if __name__ == '__main__':
	main()