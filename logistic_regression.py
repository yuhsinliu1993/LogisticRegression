import argparse
import numpy as np
from data_generator import GaussianNoise

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
    z: A scalar or numpy array of any size.
    """
    return 1.0 / (1.0 + np.exp(-1.0 * z))


def generate_training_data(mx1, vx1, my1, vy1, mx2, vx2, my2, vy2, num):
    # D1
    x1 = np.array([GaussianNoise.boxmuller(mx1, vx1) for i in range(num)])
    y1 = np.array([GaussianNoise.boxmuller(my1, vy1) for i in range(num)])

    # D2
    x2 = np.array([GaussianNoise.boxmuller(mx2, vx2) for i in range(num)])
    y2 = np.array([GaussianNoise.boxmuller(my2, vy2) for i in range(num)])

    X_train = np.vstack((np.hstack((x1, x2)), np.hstack((y1, y2))))
    X_train = np.vstack((X_train, np.ones(num*2)))      # shape = (3, 500*2) = (features, num_data)
    Y_train = np.hstack((np.zeros(num), np.ones(num))).reshape(1, -1)

    return X_train, Y_train


def print_confusion_matrix(y_preds, y):
    TP = TN = FP = FN = 0

    N = y.shape[0]

    for i in range(N):
        if y_preds[i] == 0:
            if y[i] == 0:
                TN += 1
            else:
                FN += 1
        else:
            if y[i] == 1:
                TP += 1
            else:
                FP += 1

    print("\n-------- Confusion Matrix --------")
    print("True Positive:", TP)
    print("True Negative:", TN)
    print("False Positive:", FP)
    print("False Negative:", FN)
    print("Sensitivity:", TP / (TP + FN))
    print("Specificity:", TN / (TN + FP))


class LogisticRegression(object):

    def __init__(self, N, X_train, Y_train):
        """
        N: number of total data
        X_train: data with shape (num_features, num_data)
        Y_train: true label (0: D1, 1: D2) with shape (1, number of examples)
        """
        self.N = N
        self.X_train = X_train
        self.Y_train = Y_train

        # initialize parameters
        self.w = np.zeros(shape=(X_train.shape[0], 1), dtype=np.float32)  # shape = (num_features, 1)

    def train(self, learning_rate, iteration_num, print_cost=False):

        for i in range(iteration_num):
            predictions = sigmoid(np.dot(self.w.T, self.X_train))
            cost = (-1.0) * np.mean(np.multiply(self.Y_train, np.log(predictions)) + np.multiply(1.0-self.Y_train, np.log(1.0 - predictions)), axis=1)
            cost = np.squeeze(cost)

            D = np.zeros((self.N, self.N))
            for i in range(self.N):
                D[i, i] = predictions[0, i] * (1 - predictions[0, i])
            Hessian = np.dot(np.dot(self.X_train, D), self.X_train.T)

            # Gradients
            d_w = (1.0 / self.N) * np.matmul(self.X_train, np.transpose(predictions - self.Y_train)) # (3, 1)

            if np.linalg.det(Hessian) == 0:
                # print("SGD")
                # Update the parameters
                self.w = self.w - learning_rate * d_w
            else:
                # print("Newton's method")
                self.w = self.w - np.dot(np.linalg.inv(Hessian), d_w)

            if print_cost and i % 200 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))


    def predict(self, X):
        """
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        """

        Y_pred = np.zeros(self.N)
        predictions = sigmoid(np.dot(self.w.T, X))  # (1, num_features) x (num_features, N)

        for i in range(self.N):
            Y_pred[i] = 1 if predictions[0, i] > 0.5 else 0

        return Y_pred



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, help='specify the number of data', default=500)
    parser.add_argument('--mx1', type=float, help='specify the number of data', default=0)
    parser.add_argument('--my1', type=float, help='specify the number of data', default=0)
    parser.add_argument('--vx1', type=float, help='specify the number of data', default=2)
    parser.add_argument('--vy1', type=float, help='specify the number of data', default=1)
    parser.add_argument('--mx2', type=float, help='specify the number of data', default=1.5)
    parser.add_argument('--my2', type=float, help='specify the number of data', default=1.5)
    parser.add_argument('--vx2', type=float, help='specify the number of data', default=1)
    parser.add_argument('--vy2', type=float, help='specify the number of data', default=1)
    parser.add_argument('--learning_rate', type=float, help='specify the number of data', default=5e-2)
    parser.add_argument('-i', '--iteration_num', type=int, help='specify the number of data', default=2000)

    args = parser.parse_args()

    X, Y = generate_training_data(args.mx1, args.my1, args.vx1, args.vy1, args.mx2, args.my2, args.vx2, args.vy2, args.num)

    model = LogisticRegression(X.shape[1], X, Y)

    model.train(learning_rate=args.learning_rate, iteration_num=args.iteration_num, print_cost=True)

    print("\nw: ", np.squeeze(model.w[:-1]))
    print("b: ", np.squeeze(model.w[-1]))

    Y_predictions = model.predict(X)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_predictions - np.squeeze(Y))) * 100))

    print_confusion_matrix(Y_predictions, np.squeeze(Y))
