from model import Model

from handlers.csi46traces import CSI46Traces
from handlers.csiar import CSIAR
from handlers.wiar import WiAR
from handlers.falldefi import FallDeFi
from handlers.newintel import NewIntel
from handlers.newpi import NewPi

#TEST IMPORTS
from handlers.read_pcap import BeamformReader
from handlers.matlab import db
from numpy import inf

import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

class Experiment():

    def __init__(self, dataset, config, resultsFile="results.json"):
        
        if dataset == "csiar":
            data = CSIAR(config[dataset]["directory"], config[dataset]["config"])
        elif dataset == "wiar":
            data = WiAR(config[dataset]["directory"], config[dataset]["config"])
        elif dataset == "csi46traces":
            data = CSI46Traces(config[dataset]["directory"], config[dataset]["config"])
        elif dataset == "falldefi":
            data = FallDeFi(config[dataset]["directory"], config[dataset]["config"])
        elif dataset == "newintel":
            data = NewIntel(config[dataset]["directory"], config[dataset]["config"])
        elif dataset == "newpi":
            data = NewPi(config[dataset]["directory"], config[dataset]["config"])

        self.x_all, self.y_all = data.x_all, data.y_all
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.x_all, self.y_all, stratify=self.y_all, test_size=0.3, random_state=42)

        print("x_all: " + str(self.x_all.shape))
        print("y_all: " + str(self.y_all.shape))

        batch_size = config[dataset]["config"]["batch_size"]
        epochs = config[dataset]["config"]["epochs"]

        f1s = []
        confs = []

        y_valid_total = np.array([])
        y_pred_total = np.array([])

        # n_splits = 2
        # for train_index, test_index in StratifiedKFold(n_splits, shuffle=True).split(self.x_all, self.y_all.argmax(1)):
        #     x_train, x_test = self.x_all[train_index], self.x_all[test_index]
        #     y_train, y_test = self.y_all[train_index], self.y_all[test_index]

        #     print(x_train.shape)
        #     print(x_test.shape)
        #     print(y_train.shape)
        #     print(y_test.shape)

        #     model = Model(config[dataset]["config"], structure="deepconvlstm").model
        #     (conf, classification, f1, y_valid_single, y_pred_single) = self.runExperiment(x_train, y_train, x_test, y_test, batch_size, epochs, model)
            
        #     y_valid_total = np.concatenate((y_valid_total, y_valid_single))
        #     y_pred_total = np.concatenate((y_pred_total, y_pred_single))

        #     f1s.append(f1)
        #     confs.append(conf)

        model = Model(config[dataset]["config"], structure="deepconvlstm").model
        (conf, classification, f1, y_valid_single, y_pred_single) = self.runExperiment(self.x_train, self.y_train, self.x_valid, self.y_valid, batch_size, epochs, model)
            
        # totconf = np.array([
        #     [517, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 265, 158, 1, 0, 0, 0, 1, 0, 0, 0],
        #     [0, 122, 321, 0, 0, 0, 0, 3, 0, 0, 0],
        #     [0, 0, 1, 436, 0, 1, 0, 1, 34, 0, 0],
        #     [0, 0, 0, 0, 477, 0, 0, 1, 4, 0, 0],
        #     [0, 0, 0, 0, 0, 527, 0, 0, 0, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 475, 1, 0, 0, 0],
        #     [0, 0, 3, 0, 0, 0, 0, 318, 0, 0, 0],
        #     [0, 0, 0, 3, 39, 0, 0, 0, 179, 0, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 516, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 498],
        # ])

        # print(classification_report(y_valid_total, y_pred_total, target_names=["nothing", "standup", "sitdown", "getintobed", "cook", "washingdishes", "brushteeth", "drink", "petcat", "sleeping", "walk"]))

        # totconf = np.sum(confs, axis=0)

        # ax = plt.subplot()
        # sns.heatmap(totconf, annot=True, ax=ax, fmt="g", cmap="Blues", annot_kws={"size": 16}, cbar=False)

        # ax.set_xlabel("Predicted")
        # ax.set_ylabel("True")

        # ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 15)
        # ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 15)

        # # ax.set_title("Confusion Matrix")

        # ax.xaxis.set_ticklabels(["nothing", "standup", "sitdown", "intobed", "cook", "washdish", "brushteeth", "drink", "petcat", "sleeping", "walk"])
        # ax.yaxis.set_ticklabels(["nothing", "standup", "sitdown", "intobed", "cook", "washdish", "brushteeth", "drink", "petcat", "sleeping", "walk"])

        # plt.yticks(rotation=0)

        # for tick in ax.yaxis.get_major_ticks():
        #     tick.label.set_verticalalignment('center')

        # for tick in ax.xaxis.get_major_ticks():
        #     # tick.label.set_verticalalignment('center')
        #     tick.label.set_rotation("vertical")

        # plt.show()

    def runExperiment(self, x_train, y_train, x_valid, y_valid, batch_size, epochs, model):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.summary()
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size, 
            epochs=epochs,
            shuffle=False,
            # shuffle=True
            validation_data=(x_valid, y_valid)
        )

        # model = tf.keras.models.load_model("checkpoints/model.h5")

        y_pred = model.predict(x_valid)

        y_valid_single = np.argmax(y_valid, axis=1)
        y_pred_single = np.argmax(y_pred, axis=1)

        # model.save("checkpoints/model.h5")

        print(confusion_matrix(y_valid_single, y_pred_single))
        print(classification_report(y_valid_single, y_pred_single, target_names=["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]))
        # print(classification_report(y_valid_single, y_pred_single, target_names=["nothing", "standup", "sitdown", "getintobed", "cook", "washingdishes", "brushteeth", "drink", "petcat", "sleeping", "walk"]))

        conf = confusion_matrix(y_valid_single, y_pred_single)
        classification = classification_report(y_valid_single, y_pred_single)
        f1 = f1_score(y_valid_single, y_pred_single, average="macro")

        exit(0)

        ################################

        # bfFile = BeamformReader("tests/brushteeth_1590158153.pcap")
        # csi_trace = bfFile.csi_trace

        # features = []
        # for entry in csi_trace:
        #     csiData = entry["csi"]

        #     reshapedData = np.zeros((256, 1), dtype=np.complex)
        #     for x in range(256):
        #         reshapedData[x] = csiData[x]

        #     reshapedData = db(np.abs(reshapedData))
        #     features.append(reshapedData)

        # features = np.array(features)

        # windowSize = 100
        # step = 50

        # index = 0
        # positiveInput = []
        # while index + windowSize <= features.shape[0]:
        #     curFeature = np.zeros((1, windowSize, 256))
        #     curFeature[0] = features[index:index+windowSize, :, 0]
        #     positiveInput.append(curFeature)
        #     index += step

        # positiveInput = np.concatenate(positiveInput, axis=0)
        # positiveInput[positiveInput == -inf] = 0 

        # ALL_ACTIVITIES = ["nothing", "standup", "sitdown", "getintobed", "cook", "washingdishes", "brushteeth", "drink", "petcat", "sleeping", "walk"]
        # # Perform predictions on windows.
        # y_pred = model.predict(positiveInput)
        # y_pred = np.argmax(y_pred, axis=1)

        # print([ALL_ACTIVITIES[x] for x in y_pred])

        # return (conf, classification, f1, y_valid_single, y_pred_single)