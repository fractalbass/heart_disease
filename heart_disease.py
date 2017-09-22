from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras import optimizers
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K


class HeartDisease:

    # create model
    model = Sequential()
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    max_rows = 0
    max_cols = 0
    filename = ""

    def load_dataset(self, filename):
        self.filename = filename
        dataset = np.loadtxt(filename, delimiter=",")
        self.max_rows = dataset.shape[0]
        self.max_cols = dataset.shape[1]
        return dataset

    def preprocess(self, dataset):
        dataset[:, 0:self.max_rows] = self.scaler.fit_transform(dataset[:, 0:self.max_rows])
        np.savetxt("./data/transformed_data.csv", dataset , delimiter=",")
        return dataset

    def split_dataset(self, dataset):
        X = dataset[:, 0:self.max_cols-1]
        Y = dataset[:, self.max_cols-1]
        return X,Y

    def create_model(self):

        self.model.add(Dense((self.max_cols -1) * 2, input_dim=self.max_cols-1, kernel_initializer='random_uniform', activation='tanh'))
        self.model.add(Dense(200, kernel_initializer='random_uniform', activation='tanh'))
        self.model.add(Dense(200, kernel_initializer='random_uniform', activation='tanh'))
        self.model.add(Dense(200, kernel_initializer='random_uniform', activation='tanh'))
        self.model.add(Dense(1, kernel_initializer='random_uniform', activation='sigmoid'))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    def train(self, X, Y):
        # Fit the model
        print("Training started...")
        history = self.model.fit(X, Y, validation_split=0.1, epochs=100, batch_size=25, verbose=0)
        print("Training complete.")
        return history

    def evaluate_model(self, X, Y):
        # evaluate the model
        scores = self.model.evaluate(X, Y)
        print("\n\nTraining Complete...")
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        return scores

    def plot_training_info(self, history):
        # Plot stuff to make us feel good.
        plt.plot(history.history['loss'])
        if 'val_loss' in history.history.keys:
            plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def predict(self, params):
        result = self.model.predict(params)
        return result

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = load_model(filename)

    def full_run(self):
        # fix random seed for reproducibility
        np.random.seed(7)

        # Load dataset
        dataset = self.load_dataset("./data/pima_indians_diabetes.csv")

        # # Scale Dataset
        # dataset = self.preprocess(dataset)

        # split into input (X) and output (Y) variables
        X, Y = self.split_dataset(dataset)

        # Build the model
        self.create_model()

        # Train the model
        history = self.train(X, Y)

        # Evaluate the model
        self.evaluate_model(X, Y)

        # Plot the training info
        self.plot_training_info(history)

        # Save the network
        self.save_model("pima_model")

    def get_binary_category(self, v):
        if v > .5:
            return 1
        else:
            return -0

    def get_five_class_category(self, v):
        if v > .75:
            return 1
        if v > .25:
            return 0.5
        if v > -.25:
            return 0
        if v > -.75:
            return -0.5
        return -1

if __name__ == "__main__":
    hd = HeartDisease()
    hd.full_run()

    # Load a model (We can comment out the above line if we have already trained a model)
    hd2 = HeartDisease()
    hd2.load_model("pima_model")
    a = hd2.load_dataset("./data/pima_indians_diabetes_validation.csv")

    prediction = hd2.predict(a[:,0:hd2.max_cols-1])
    correct = 0
    total_records = len(prediction)
    for p in range(0, total_records):
        pred_category = hd2.get_binary_category(prediction[p])
        actual_category = hd2.get_binary_category(a[p, hd2.max_cols-1])
        print (prediction[p], pred_category, actual_category)
        if pred_category  == actual_category:
            correct = correct + 1

    accuracy = (correct / total_records) * 100

    print("The results of the prediction are: ")
    print("Total Correct: {0} out of {1}   Accuracy: {2:.5f}%".format(correct, total_records, accuracy))
    print("Complete.")