from tensorflow import keras 
from tensorflow.keras import layers , callbacks , activations , optimizers ,losses

def model(X_train_seq_pad):
    model= keras.Sequential([
        layers.Input(shape=X_train_seq_pad.shape[1:]),
        layers.Embedding(input_dim=5001,output_dim=16),
        layers.GRU(64,dropout=0.2),
        layers.BatchNormalization(),
        layers.Dense(32,activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(5,activation='softmax')
        
    ])
    return model 


