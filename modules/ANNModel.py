import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns


class ANNModel:
    """
    A class for training and evaluating an artificial neural network (ANN) model for classification tasks.

    Args:
        n_classes (int): The number of classes in the classification task.
        X_train (numpy.ndarray): The training features.
        X_test (numpy.ndarray): The testing features.
        y_train (numpy.ndarray): The training target.
        y_test (numpy.ndarray): The testing target.
        path (str): The directory path for saving the hyperparameter tuning results.
        folder (str): The name of the folder for saving the hyperparameter tuning results.
        early_stopping (keras.callbacks.EarlyStopping): The early stopping callback.

    Methods:
        build_model(hp): Build the ANN model with hyperparameters.
        obtain_best_model(path, folder): Obtain the best model using random search hyperparameter tuning.
        train_and_predict(): Train the model and predict labels for the test data.
        evaluate(): Evaluate the model on the test data.
        scores(): Calculate and print various evaluation metrics.
        figures(): Plot ROC curves and the confusion matrix.

    Usage:
        ann_model = ANNModel(n_classes, X_train, X_test, y_train, y_test, path, folder, early_stopping)
        ann_model.train_and_predict()
        ann_model.evaluate()
        ann_model.scores()
        ann_model.figures()
    """

    def __init__(self, n_classes, X_train, X_test, y_train, y_test, folder, early_stopping):
        """
        Initialize the ANNModel object.

        Args:
            n_classes (int): The number of classes in the classification task.
            X_train (numpy.ndarray): The training features.
            X_test (numpy.ndarray): The testing features.
            y_train (numpy.ndarray): The training target.
            y_test (numpy.ndarray): The testing target.
            path (str): The directory path for saving the hyperparameter tuning results.
            folder (str): The name of the folder for saving the hyperparameter tuning results.
            early_stopping (keras.callbacks.EarlyStopping): The early stopping callback.
        """
        self.n_classes = n_classes
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.early_stopping = early_stopping
        self.model = self.obtain_best_model(folder)
        self.y_pred = None
        self.y_pred_labels = None

    def build_model(self, hp):
        """
        Build the ANN model with hyperparameters.

        Args:
            hp (kerastuner.HyperParameters): The hyperparameters object.

        Returns:
            keras.models.Sequential: The built ANN model.
        """
        model = Sequential()
        for i in range(hp.Int("num_layers", 1, 5)):
            model.add(
                Dense(
                    # Tune number of units separately.
                    units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )
        model.add(Dense(self.n_classes, activation='softmax'))
        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
        
        return model

    def obtain_best_model(self, folder):
        """
        Obtain the best model using random search hyperparameter tuning.

        Args:
            path (str): The directory path for saving the hyperparameter tuning results.
            folder (str): The name of the folder for saving the hyperparameter tuning results.

        Returns:
            keras.models.Sequential: The best model obtained from hyperparameter tuning.
        """
        tuner = RandomSearch(
            self.build_model,
            objective='val_accuracy',
            max_trials=10,
            executions_per_trial=5,
            project_name=folder
            )

        tuner.search(self.X_train,
                     self.y_train - 1,
                     epochs=200,
                     validation_data=(self.X_test, self.y_test - 1),
                     batch_size=500,
                     callbacks=[self.early_stopping]
                     )

        tuner.results_summary()

        best_hps = tuner.get_best_hyperparameters()[0]
        model = tuner.hypermodel.build(best_hps)

        return model

    def train_and_predict(self):
        """
        Train the model and predict labels for the test data.
        """
        self.model.fit(self.X_train, self.y_train - 1, validation_data=(self.X_test, self.y_test - 1), epochs=200, batch_size=500)

        # Use the trained model to predict labels for the test data
        self.y_pred = self.model.predict(self.X_test)

        # Convert the predicted labels from probabilities to class labels
        self.y_pred_labels = np.argmax(self.y_pred, axis=1) + 1

    def evaluate(self):
        """
        Evaluate the model on the test data.
        """
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test - 1)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")

    def scores(self):
        """
        Calculate and print various evaluation metrics.
        """
        # Calculate the accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred_labels)

        # Calculate precision, recall, F1-score, R2 score, MSE, RMSE, and MAE
        precision = precision_score(self.y_test, self.y_pred_labels, average='macro', zero_division=0)
        recall = recall_score(self.y_test, self.y_pred_labels, average='macro', zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred_labels, average='macro', zero_division=0)

        # Print the evaluation metrics
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)

    def figures(self):
        """
        Plot ROC curves and the confusion matrix.
        """
        # Calculate the ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.n_classes):
            # Check if there are positive samples for the current class
            if any(self.y_test == i+1):
                fpr[i], tpr[i], _ = roc_curve((self.y_test == i+1), self.y_pred[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            else:
                fpr[i] = []
                tpr[i] = []
                roc_auc[i] = 0.0

        # Plot the ROC curves
        plt.figure(figsize=(8, 6))

        for i in range(self.n_classes):
            plt.plot(fpr[i], tpr[i], label=f'Index {i + 1} (AUC = {roc_auc[i]:.5f})')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.show()

        # Define the labels for the confusion matrix
        labels = range(1, self.n_classes + 1)

        # Calculate the confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred_labels, labels=labels)

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
