"""
finetune.py
Improves the models using YOUR handwriting (saved from the GUI "Teach" box).

How it works:
  - mixes your samples (repeated many times) with some MNIST images
  - trains every model in models/ for a few more epochs with a small learning rate
  - MNIST images stay in the mix so the models don't forget normal digits

Run:  python finetune.py      (then restart gui.py)
"""
import os
import glob
import numpy as np
from tensorflow import keras

from model_def import load_mnist, make_train_dataset

USER_FILE = os.path.join("user_data", "samples.npz")
REPEAT = 20          # how many times your samples are repeated in training
MNIST_SUBSET = 15000
EPOCHS = 5

if not os.path.exists(USER_FILE):
    raise SystemExit("No samples yet. Use the 'Teach' box in gui.py to save some first.")

data = np.load(USER_FILE)
xu = data["x"].astype("float32")[..., np.newaxis]
yu = data["y"].astype("int64")
print(f"Your samples: {len(yu)}  (per digit: {np.bincount(yu, minlength=10).tolist()})")

rng = np.random.default_rng(0)

# keep 20% of your samples aside to measure improvement (only if you have enough)
if len(yu) >= 50:
    idx = rng.permutation(len(yu))
    n_hold = len(yu) // 5
    hold, tr = idx[:n_hold], idx[n_hold:]
    xu_hold, yu_hold = xu[hold], yu[hold]
    xu_tr, yu_tr = xu[tr], yu[tr]
else:
    xu_hold = yu_hold = None
    xu_tr, yu_tr = xu, yu

x_train, y_train, x_test, y_test = load_mnist()
sub = rng.choice(len(x_train), MNIST_SUBSET, replace=False)

x_mix = np.concatenate([x_train[sub]] + [xu_tr] * REPEAT)
y_mix = np.concatenate([y_train[sub]] + [yu_tr] * REPEAT)
print("Training set size:", len(y_mix))

for path in sorted(glob.glob("models/*.keras")):
    print(f"\n--- Fine-tuning {path} ---")
    model = keras.models.load_model(path)
    model.compile(optimizer=keras.optimizers.Adam(3e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    if xu_hold is not None:
        print("Your handwriting accuracy BEFORE:",
              f"{model.evaluate(xu_hold, yu_hold, verbose=0)[1] * 100:.1f}%")

    model.fit(make_train_dataset(x_mix, y_mix), epochs=EPOCHS, verbose=2)

    print("MNIST test accuracy:", f"{model.evaluate(x_test, y_test, verbose=0)[1] * 100:.2f}%")
    if xu_hold is not None:
        print("Your handwriting accuracy AFTER: ",
              f"{model.evaluate(xu_hold, yu_hold, verbose=0)[1] * 100:.1f}%")
    model.save(path)

print("\nDone. Restart gui.py to use the improved models.")
