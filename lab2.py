import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Training Shape:", x_train.shape)
print("Testing Shape :", x_test.shape)

class_names = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]


plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(y_train, bins=np.arange(11) - 0.5)
plt.xticks(range(10), class_names, rotation=45)
plt.title("Class Distribution")
plt.show()

x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print("Flattened Shape:", x_train.shape)

baseline_model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

baseline_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

baseline_history = baseline_model.fit(
    x_train,
    y_train_cat,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)


baseline_pred = np.argmax(
    baseline_model.predict(x_test),
    axis=1
)

baseline_acc = accuracy_score(y_test, baseline_pred)

print("\nBaseline Accuracy:", baseline_acc)

plt.plot(baseline_history.history['accuracy'])
plt.title("Training Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()


plt.plot(baseline_history.history['val_accuracy'])
plt.title("Validation Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.plot(baseline_history.history['loss'])
plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


plt.plot(baseline_history.history['val_loss'])
plt.title("Validation Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


configs = [
    {"neurons": 64, "lr": 0.001},
    {"neurons": 128, "lr": 0.001},
    {"neurons": 256, "lr": 0.001},
    {"neurons": 128, "lr": 0.0001},
]

results = []

best_acc = 0
best_model = None
best_config = None

for config in configs:

    print("\nTesting:", config)

    model = Sequential([
        Dense(config["neurons"],
              activation='relu',
              input_shape=(784,)),
        Dense(config["neurons"] // 2,
              activation='relu'),
        Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["lr"]
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train_cat,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    val_acc = max(history.history['val_accuracy'])

    results.append(val_acc)

    print("Validation Accuracy:", val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_model = model
        best_config = config

print("\nBest Configuration")
print(best_config)
print("Best Validation Accuracy:", best_acc)


plt.plot(results, marker='o')
plt.title("Hyperparameter Search Results")
plt.xlabel("Configuration Number")
plt.ylabel("Validation Accuracy")
plt.show()

optimized_pred = np.argmax(
    best_model.predict(x_test),
    axis=1
)

optimized_acc = accuracy_score(
    y_test,
    optimized_pred
)

optimized_precision = precision_score(
    y_test,
    optimized_pred,
    average='weighted'
)

optimized_recall = recall_score(
    y_test,
    optimized_pred,
    average='weighted'
)

optimized_f1 = f1_score(
    y_test,
    optimized_pred,
    average='weighted'
)

print("\nOptimized Accuracy:", optimized_acc)
print("Precision:", optimized_precision)
print("Recall:", optimized_recall)
print("F1 Score:", optimized_f1)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    optimized_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    xticks_rotation=45
)

plt.title("Confusion Matrix")
plt.show()

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report")
print(
    classification_report(
        y_test,
        optimized_pred
    )
)

# =====================================================
# ACCURACY COMPARISON
# =====================================================

plt.bar(
    ["Baseline", "Optimized"],
    [baseline_acc, optimized_acc]
)

plt.ylabel("Accuracy")
plt.title("Best Model Accuracy Comparison")
plt.show()

# =====================================================
# PERFORMANCE TABLE
# =====================================================

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Baseline": [
        baseline_acc,
        precision_score(
            y_test,
            baseline_pred,
            average='weighted'
        ),
        recall_score(
            y_test,
            baseline_pred,
            average='weighted'
        ),
        f1_score(
            y_test,
            baseline_pred,
            average='weighted'
        )
    ],
    "Optimized": [
        optimized_acc,
        optimized_precision,
        optimized_recall,
        optimized_f1
    ]
})

print("\nPerformance Comparison")
print(comparison)