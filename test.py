from handlers.read_pcap import BeamformReader
from handlers.matlab import db

from numpy import inf

import tensorflow as tf
import numpy as np
import os
import sys

tf.keras.backend.clear_session()

ALL_ACTIVITIES = ["nothing", "standup", "sitdown", "getintobed", "cook", "washingdishes", "brushteeth", "drink", "petcat", "sleeping", "walk"]

model = tf.keras.models.load_model("checkpoints/model.h5")

bfFile = BeamformReader("tests/washingdishes_1591633721.pcap")
csi_trace = bfFile.csi_trace

features = []
for entry in csi_trace:
    csiData = entry["csi"]

    reshapedData = np.zeros((256, 1), dtype=np.complex)
    for x in range(256):
        reshapedData[x] = csiData[x]

    reshapedData = db(np.abs(reshapedData))
    features.append(reshapedData)

features = np.array(features)

windowSize = 100
step = 50

index = 0
positiveInput = []
while index + windowSize <= features.shape[0]:
    curFeature = np.zeros((1, windowSize, 256))
    curFeature[0] = features[index:index+windowSize, :, 0]
    positiveInput.append(curFeature)
    index += step

positiveInput = np.concatenate(positiveInput, axis=0)
positiveInput[positiveInput == -inf] = 0 

# Perform predictions on windows.
y_pred = model.predict(positiveInput)
print(y_pred)

y_pred = np.argmax(y_pred, axis=-1)
print(y_pred)

print([ALL_ACTIVITIES[x] for x in y_pred])