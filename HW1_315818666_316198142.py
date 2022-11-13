import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """

        self.k = k
        self.p = p
        self.train_df = pd.DataFrame()
        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (315818666, 316198142)

    def find_distance(self, x1, x2, p):
        """
        This method calculates the distance between two vectors.

        :param x1: A vector.
        :param x2: A vector.
        :param p: A parameter for minkowski distance calculation.
        :return A dataframe of the k nearest neighbors for a specific sample according to distance and then label.
        """

        dist = np.linalg.norm(np.array(x1) - np.array(x2), ord=p)
        return dist


    def k_neighbors(self, test_sample):
        """
        This method finds the k nearest neighbors for a specific sample.

        :param test_sample: A list of a specific sample from the test set.
        :return A dataframe of the k nearest neighbors for a specific sample according to distance and then label.
        """

        train_dataframe = self.train_df.copy()
        train_dataframe['distance'] = train_dataframe.apply(lambda c: self.find_distance(c.point, test_sample, self.p),
                                                            axis=1)
        # sorting
        train_dataframe = train_dataframe.sort_values(by=["distance", "label"],
                                                      ascending=[True, True]).reset_index(drop=True)
        # choosing the k neighbors
        k_nn = train_dataframe.head(self.k)
        return k_nn


    def predict_specific_sample(self, test_sample):
        """
        This method predicts the best label match for a specific sample.

        :param test_sample: A list of a specific sample from the test set.
        :return the best label match for a specific sample (by distance and then by the label).
        """

        # finding the knn
        k_nn = KnnClassifier.k_neighbors(self, test_sample)

        # count for each label
        count = k_nn.groupby('label').size().reset_index()
        count = count.rename(columns={'label': 'label', 0: 'count'})
        sorted_knn = count.sort_values(by=["count"], ascending=[False]).reset_index(drop=True)

        # finding the best labels
        count_filter = count['count'] == sorted_knn.iloc[0]['count']
        k_nn_best_labels = count[count_filter]
        best_labels = k_nn_best_labels['label'].tolist()

        # get the samples with the best labels in the knn
        best_labels_filter = k_nn['label'].isin(best_labels)
        best_labels_samples = k_nn[best_labels_filter]
        # sorting
        best_labels_samples = best_labels_samples.sort_values(by=["distance", "label"],
                                                                ascending=[True, True]).reset_index(drop=True)
        # choosing the best label
        best_label = best_labels_samples.iloc[0]['label']
        return best_label


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        
        point_label = {'point': X.tolist(), 'label': y.tolist()}
        self.train_df = pd.DataFrame(point_label, columns=['point', 'label'])
        pass


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """

        list_of_x = X.tolist()
        test_dataframe = pd.DataFrame({'point': list_of_x}, columns=['point'])

        test_dataframe['label'] = test_dataframe.apply(lambda c: KnnClassifier.predict_specific_sample(self, c.point),
                                                       axis=1)
        # convert to numpy array
        test_array = test_dataframe['label'].to_numpy()

        return test_array
        pass


        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)


def main():

    print("*" * 20)
    print("Started HW1_315818666_316198142.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()
