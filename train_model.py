"""
train_model.py  (version 2)
Trains an ENSEMBLE of CNNs on MNIST (their predictions are averaged = higher accuracy).
Saves models/digit_model_0.keras, _1, _2 ...
Also saves training_curves.png and confusion_matrix.png for your report.

Takes longer than before. To make it faster, reduce N_MODELS or EPOCHS.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

from model_def import build_model, load_mnist, make_train_dataset

N_MODELS = 3
EPOCHS = 12

os.makedirs("models", exist_ok=True)

x_train, y_train, x_test, y_test = load_mnist()
x_val, y_val = x_train[-5000:], y_train[-5000:]       # 5000 images kept for validation
x_tr, y_tr = x_train[:-5000], y_train[:-5000]
print("Train:", x_tr.shape, " Validation:", x_val.shape, " Test:", x_test.shape)

all_probs = []
last_history = None

for i in range(N_MODELS):
    print(f"\n===== Training model {i + 1} of {N_MODELS} =====")
    keras.utils.set_random_seed(100 + i)               # different start for each model

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                          patience=2, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                      restore_best_weights=True),
    ]
    last_history = model.fit(make_train_dataset(x_tr, y_tr),
                             epochs=EPOCHS,
                             validation_data=(x_val, y_val),
                             callbacks=callbacks)

    acc = model.evaluate(x_test, y_test, verbose=0)[1]
    print(f"Model {i + 1} test accuracy: {acc * 100:.2f}%")

    model.save(f"models/digit_model_{i}.keras")
    all_probs.append(model.predict(x_test, verbose=0))

# ---------- Ensemble result ----------
ens_probs = np.mean(all_probs, axis=0)
y_pred = np.argmax(ens_probs, axis=1)
ens_acc = np.mean(y_pred == y_test)
print(f"\nENSEMBLE test accuracy: {ens_acc * 100:.2f}%")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# ---------- Graphs for the report ----------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(last_history.history["accuracy"], label="Train")
plt.plot(last_history.history["val_accuracy"], label="Validation")
plt.title("Accuracy (last model)")
plt.xlabel("Epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(last_history.history["loss"], label="Train")
plt.plot(last_history.history["val_loss"], label="Validation")
plt.title("Loss (last model)")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix (ensemble, {ens_acc * 100:.2f}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.xticks(range(10))
plt.yticks(range(10))
for r in range(10):
    for c in range(10):
        plt.text(c, r, cm[r, c], ha="center", va="center", fontsize=7,
                 color="white" if cm[r, c] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved models/ folder, training_curves.png, confusion_matrix.png")
