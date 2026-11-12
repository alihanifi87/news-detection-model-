# Text Classification with GRU

A deep learning project built with TensorFlow/Keras and Scikit-learn to classify text data into 5 distinct categories using a Gated Recurrent Unit (GRU) neural network.

## 📁 Project Structure

```text
├── load_dataset.py   # Utility script for loading the CSV dataset
├── model.py          # Defines the Keras Sequential GRU model architecture
├── main.py           # Main script for text preprocessing, tokenization, training, and evaluation
└── text_type.csv     # Dataset file (expected in the root directory)
```

## 🛠️ Prerequisites & Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn tensorflow
```

## 📊 Dataset Format

The pipeline expects a CSV file named `text_type.csv` in the root directory containing at least two columns:
* `Text`: The raw text samples to be classified.
* `Label`: Integer-encoded target labels (0 through 4, corresponding to the 5 output classes).

## 🧠 Model Architecture

The neural network defined in `model.py` uses the following architecture:
1. **Input Layer**: Accepts padded integer sequences of length `100`.
2. **Embedding Layer**: Maps a vocabulary of up to 5,000 words into 16-dimensional dense vectors (`input_dim=5001`, `output_dim=16`).
3. **GRU Layer**: 64 units with a dropout rate of `0.2` to capture sequential dependencies in the text.
4. **Batch Normalization**: Stabilizes and accelerates training.
5. **Dense Layer**: 32 units with `ReLU` activation, followed by a `0.2` Dropout layer for regularization.
6. **Output Layer**: 5 units with `Softmax` activation for multi-class probability distribution.

## 🚀 Usage

1. Place your `text_type.csv` dataset in the same directory as the Python scripts.
2. Run the main pipeline:

```bash
python main.py
```

## ⚙️ Pipeline Overview

* **Text Cleaning**: Strips punctuation and special characters using regular expressions.
* **Data Splitting**: Splits the dataset into 80% training and 20% testing sets (`random_state=1`).
* **Tokenization & Padding**: Tokenizes the top 5,000 most frequent words and post-pads/truncates sequences to a maximum length of `100`.
* **Training Configuration**: Compiled with the `Adam` optimizer (`lr=0.001`) and `SparseCategoricalCrossentropy` loss, utilizing `EarlyStopping` and `ReduceLROnPlateau` callbacks.

## 📝 Notes & Known Issues in Current Code

If you are actively developing `main.py`, keep the following in mind:
* **Model Instantiation**: In `main.py`, `model` is imported as a function (`from model import model`), so it must be instantiated with `m = model(X_train_seq_pad)` before calling `m.compile()`.
* **Training Step**: Ensure `model.fit(...)` is called with your training data and callbacks (`es`, `rlr`) prior to running `model.predict()`.
* **Evaluation Metric**: Since the model outputs a 5-class softmax probability distribution, consider using `np.argmax(y_pred, axis=1)` alongside `accuracy_score` or `f1_score` rather than `mean_absolute_error`.