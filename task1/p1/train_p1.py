from p1 import create_model, load_data, save_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import joblib
from parameters import *


# create these folders if they does not exist within p1 directory
if not os.path.isdir("p1"):
    os.mkdir("p1")

if not os.path.isdir("p1/results"):
    os.mkdir("p1/results")

if not os.path.isdir("p1/logs"):
    os.mkdir("p1/logs")

if not os.path.isdir("p1/data"):
    os.mkdir("p1/data")

# create model checkpoints directory within p1
if not os.path.isdir("p1/model_checkpoints"):
    os.mkdir("p1/model_checkpoints")

# load the data using yfinance (ticker is loaded inside load_data function)
data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS)

# save the dataframe
data["df"].to_csv(ticker_data_filename)

# construct the model
model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("p1/results", model_name + ".weights.h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("p1/logs", model_name))

# train the model and save the weights whenever we see
# a new optimal model using ModelCheckpoint
history = model.fit(data["X_train"], data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)

# save the complete model in .keras format
model_checkpoint_path = os.path.join("p1/model_checkpoints", model_name + ".keras")
save_model(model, model_checkpoint_path)

# save the data dictionary for inference
data_path = os.path.join("p1/model_checkpoints", model_name + "_data.pkl")
joblib.dump(data, data_path)
print(f"Data saved to {data_path}")
