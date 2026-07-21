import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


df = pd.read_csv(
    "/Users/mmadando/Downloads/data_banknote_authentication.txt",
    header=None
)

df.columns = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
    "class"
]

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nSTATISTICS")
print(df.describe())

# Histograms
df.hist(figsize=(12, 8))
plt.suptitle("Feature Histograms")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="variance",
    y="skewness",
    hue="class"
)
plt.title("Variance vs Skewness")
plt.show()

# Boxplots
plt.figure(figsize=(10, 6))
df.drop("class", axis=1).boxplot()
plt.title("Feature Boxplots")
plt.show()


X = df.drop("class", axis=1).values
y = df["class"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAIN SIZE:", len(X_train))
print("TEST SIZE:", len(X_test))


class Perceptron:

    def __init__(self, learning_rate=0.01, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = 0

        self.errors = []

        self.weight_history = []
        self.bias_history = []

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):

        n_features = X.shape[1]

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            errors = 0

            for xi, target in zip(X, y):

                linear_output = np.dot(xi, self.weights) + self.bias

                prediction = self.activation(linear_output)

                update = self.learning_rate * (target - prediction)

                self.weights += update * xi
                self.bias += update

                if update != 0:
                    errors += 1

            self.errors.append(errors)

            self.weight_history.append(
                self.weights.copy()
            )

            self.bias_history.append(
                self.bias
            )

            print(
                f"Epoch {epoch+1:02d} | Errors = {errors}"
            )

    def predict(self, X):

        linear_output = np.dot(X, self.weights) + self.bias

        return self.activation(linear_output)


perceptron = Perceptron(
    learning_rate=0.01,
    epochs=50
)

perceptron.fit(
    X_train,
    y_train
)



y_pred = perceptron.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nFINAL WEIGHTS")
print(perceptron.weights)

print("\nFINAL BIAS")
print(perceptron.bias)

print("\nPERFORMANCE")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nCONFUSION MATRIX")
print(cm)

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(perceptron.errors)+1),
    perceptron.errors,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.title("Training Error vs Epoch")
plt.grid(True)
plt.show()

weights = np.array(
    perceptron.weight_history
)

plt.figure(figsize=(10, 6))

for i in range(weights.shape[1]):
    plt.plot(
        range(1, len(weights)+1),
        weights[:, i],
        label=f"W{i+1}"
    )

plt.xlabel("Epoch")
plt.ylabel("Weight Value")
plt.title("Weight Evolution")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(perceptron.bias_history)+1),
    perceptron.bias_history,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Bias")
plt.title("Bias Evolution")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

learning_rates = [
    0.001,
    0.01,
    0.1
]

plt.figure(figsize=(10, 6))

for lr in learning_rates:

    model = Perceptron(
        learning_rate=lr,
        epochs=50
    )

    model.fit(
        X_train,
        y_train
    )

    plt.plot(
        range(1, len(model.errors)+1),
        model.errors,
        label=f"LR={lr}"
    )

plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid(True)

plt.show()



epoch_table = pd.DataFrame({
    "Epoch":
    range(
        1,
        len(perceptron.errors)+1
    ),

    "Errors":
    perceptron.errors,

    "Bias":
    perceptron.bias_history
})

print("\nEPOCH WISE LEARNING")
print(epoch_table)

summary = pd.DataFrame({

    "Metric": [
        "Dataset Size",
        "Train/Test Split",
        "Learning Rate",
        "Epochs",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value": [
        len(df),
        "80/20",
        0.01,
        50,
        round(accuracy,4),
        round(precision,4),
        round(recall,4),
        round(f1,4)
    ]
})

print("\nTRAINING SUMMARY")
print(summary)