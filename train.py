import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load + preprocess
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # saves mnist.npz in ~/.keras/datasets
x_train = (x_train / 255.0).astype("float32").reshape(-1, 28, 28, 1)
x_test  = (x_test  / 255.0).astype("float32").reshape(-1, 28, 28, 1)

# 2. CNN
cnn = keras.Sequential([
    layers.Input((28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Flatten(), layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")])
cnn.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
h = cnn.fit(x_train, y_train, epochs=8, batch_size=128, validation_split=0.1)
loss, acc = cnn.evaluate(x_test, y_test)
print("CNN test accuracy:", acc)
cnn.save("model.keras")

# 3. MLP (optional comparison)
mlp = keras.Sequential([layers.Input((28,28,1)), layers.Flatten(),
    layers.Dense(256, activation="relu"), layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")])
mlp.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
mlp.fit(x_train, y_train, epochs=8, batch_size=128, validation_split=0.1, verbose=0)
print("MLP test accuracy:", mlp.evaluate(x_test, y_test, verbose=0)[1])

# 4. Loss curve
plt.plot(h.history["loss"], label="train"); plt.plot(h.history["val_loss"], label="val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.savefig("loss_curve.png"); plt.close()

# 5. Confusion matrix
pred = cnn.predict(x_test).argmax(1)
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.savefig("confusion_matrix.png"); plt.close()
print(classification_report(y_test, pred))

# 6. Misclassified examples
wrong = np.where(pred != y_test)[0][:10]
for i, idx in enumerate(wrong):
    plt.subplot(2, 5, i+1); plt.imshow(x_test[idx].squeeze(), cmap="gray")
    plt.title(f"T:{y_test[idx]} P:{pred[idx]}"); plt.axis("off")
plt.savefig("errors.png")