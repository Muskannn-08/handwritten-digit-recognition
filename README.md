# Handwritten Digit Recognition System

A mini project that recognizes handwritten numbers (any length, up to 8 digits) drawn on a canvas or uploaded as an image. It uses an ensemble of CNNs trained on the MNIST dataset.

## Features
- Draw a number on a canvas, or upload an image
- Works for single digits and multi-digit numbers
- Two modes: **Boxes** (most accurate) and **Free drawing** (auto-split)
- Shows confidence and top-3 predictions for each digit
- Warns when the model is not sure
- Teach the model your own handwriting and fine-tune it

## Tech Stack
Python, TensorFlow/Keras, NumPy, SciPy, Pillow, scikit-learn, Matplotlib, Tkinter

## How It Works
```
Drawing / Image
      |
Preprocessing (grayscale, clean background)
      |
Split into digits (boxes or connected components)
      |
Resize to 28x28, centre, normalize
      |
Ensemble of 3 CNNs (average of predictions)
      |
Predicted number + confidence
```

## Project Structure
| File | Purpose |
|------|---------|
| `model_def.py` | CNN architecture and data augmentation |
| `train_model.py` | Trains the CNN ensemble on MNIST |
| `predict.py` | Preprocessing, digit splitting, prediction |
| `gui.py` | Tkinter app |
| `finetune.py` | Improves the model using your saved handwriting |

## Setup and Run
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

python train_model.py        # trains models (takes a while)
python gui.py                # start the app
```

To improve accuracy on your own handwriting:
1. In the app, type the correct number in the Teach box and click **Save as training sample** (aim for 20+ per digit).
2. Run `python finetune.py`, then restart `gui.py`.

## Tips
- Draw big digits and stay inside the boxes.
- Use Boxes mode if digits touch each other.

## Results
- MNIST test accuracy (ensemble): **add your result here**
- Add `training_curves.png` and `confusion_matrix.png` from training.
