import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional # type: ignore
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import clone_model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore


import joblib

# Step 1: Import preprocessing function
from preprocess import preprocess_and_create_sequences

# File paths
divi_path = "data/DiviFinal.csv"
pranav_path = "data/PranavFinal.csv"
final_path = "data/Final.csv"



# Step 2: Load data and preprocess
X_train, X_test, y_train, y_test, label_encoder = preprocess_and_create_sequences(divi_path, pranav_path, final_path)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)


print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Label classes: {label_encoder.classes_}")

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# Build the model
# model = Sequential()
# model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(y_train_cat.shape[1], activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.4))
# model.add(LSTM(32, return_sequences=False))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(y_train_cat.shape[1], activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

# Build the model
# model = Sequential()
# model.add(Bidirectional(LSTM(128, return_sequences=False), input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Dropout(0.4))  # Slightly higher dropout for bigger LSTM
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(y_train_cat.shape[1], activation='softmax'))

# Compile with lower learning rate
optimizer = Adam(learning_rate=0.0005)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])




model = Sequential()

# 1st LSTM layer (Bidirectional, returns sequences)
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.5))  # slightly higher dropout since more params

# 2nd LSTM layer (no return_sequences now)
model.add(Bidirectional(LSTM(32, return_sequences=False)))
model.add(Dropout(0.3))

# Dense layers
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Output
model.add(Dense(y_train_cat.shape[1], activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train_cat,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stop]
)


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {accuracy:.2f}")


# Save the model
model.save("Models/lstm_model.h5")
print("Model saved to Models/lstm_model.h5")

# Save encoders
joblib.dump(label_encoder, "Models/label_encoder.pkl")

print("Encoders saved.")