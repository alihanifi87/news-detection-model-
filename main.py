import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score , confusion_matrix , f1_score,mean_absolute_error
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers , callbacks , activations , optimizers ,losses
import re
from model import model

from load_dataset import load_data

df = load_data('text_type.csv')

X=df['Text']
y=df['Label']


X_clean=[re.sub(r'[^\s\w]','',x) for x in X]
X_cleaner = [re.sub(r'[\n\t]','',x.lower()) for x in X_clean]

X_train ,X_test, y_train , y_test = train_test_split(X_clean,y , test_size=0.2,random_state=1)

tk = Tokenizer()

tk.fit_on_texts(X_train)
tk.word_index

len(tk.word_index)

tk=Tokenizer(num_words=5000)
tk.fit_on_texts(X_train)

X_train_seq= tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)

ls=[]

for i in range(len(X_train_seq)):
    ls.append(len(X_train_seq[i]))




ls.sort()
ls[int(len(ls)/2)]

X_train_seq_pad= pad_sequences(X_train_seq , padding='post', maxlen=100)
X_test_seq_pad= pad_sequences(X_test_seq , padding='post', maxlen=100)




adam =optimizers.Adam(learning_rate=0.001)
scce = losses.SparseCategoricalCrossentropy()
model.compile(
    optimizer=adam,
    loss=scce,
    metrics=['accuracy']
)

model.summary()


es=callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
rlr=callbacks.ReduceLROnPlateau(monitor='val_loss',patience=5,factor=0.5)


y_pred=model.predict(X_test_seq_pad)
print(mean_absolute_error(y_test,y_pred))





