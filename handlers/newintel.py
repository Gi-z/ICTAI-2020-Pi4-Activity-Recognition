from .handler import Handler
from .read_bfee import BeamformReader
from .matlab import db

from numpy import inf

import glob
import os

import numpy as np

class NewIntel(Handler):

    FS = 100
    ALL_ACTIVITIES = ["nothing", "standup", "sitdown", "getintobed", "cook", "washingdishes", "brushteeth", "drink", "petcat", "sleeping", "walk"]

    def __init__(self, directory, config):
        super().__init__(directory, config)

        self.initialiseData(directory, config["downsample"])
        
    def loadFromDat(self, inputFile, windowSize=FS, step=50):

        try:
            bfFile = BeamformReader(inputFile)
        except ValueError as e:
            print("Unable to parse CSI file: {}".format(inputFile))
            print(e)
            print("Skipping.")
            return None

        csi_trace = bfFile.csi_trace
        
        features = []
        for entry in csi_trace:
            csiData = entry["csi"]

            # reshapedData = np.zeros((90, 1), dtype=np.complex)
            reshapedData = np.zeros((60, 1), dtype=np.complex)
            ind = 0

            try:
                for x in range(30):
                    reshapedData[ind] = csiData[x][0][0]
                    reshapedData[ind+1] = csiData[x][1][0]
                    # reshapedData[ind+2] = csiData[x][2][0]
                    ind += 2
            except IndexError as e:
                print("invalid csi length")

                


            reshapedData = db(np.abs(reshapedData))
            features.append(reshapedData)

        features = np.array(features)
            
        index = 0
        positiveInput = []
        while index + windowSize <= features.shape[0]:
            # curFeature = np.zeros((1, windowSize, 90))
            curFeature = np.zeros((1, windowSize, 60))
            curFeature[0] = features[index:index+windowSize, :, 0]
            positiveInput.append(curFeature)
            index += step

        try:
            positiveInput = np.concatenate(positiveInput, axis=0)
        except ValueError as e:
            # positiveInput = np.zeros((1, windowSize, 90))
            positiveInput = np.zeros((1, windowSize, 60))

        positiveInput[positiveInput == -inf] = 0

        return positiveInput

    def loadByActivityLabel(self, directory, activity, activities=ALL_ACTIVITIES, saveToFile=False, windowSize=FS, step=50, downsample=1):

        compressedFilename = "x_{}_win_{}_step_{}.npz".format(activity, windowSize, step)
        compressedPath = os.path.join(directory, compressedFilename)

        if os.path.exists(compressedPath):
            print("Compressed data for activity with experiment parameters exists.")
            print("Loading: " + compressedPath)

            inputArray = np.load(compressedPath)["arr_0"]

            if downsample > 1:
                inputArray = inputArray[:, ::downsample, :]

            #Debug
            inputArray[inputArray == -inf] = 0

            outputLabels = np.zeros((inputArray.shape[0], len(activities)))
            outputLabels[:, activities.index(activity)] = 1
            return inputArray, outputLabels

        print("Loading CSI data for activity: " + activity)
        activity = activity.lower()
        if activity not in activities:
            print("Invalid activity: " + activity)
            print("Ensure a valid activity is provided for the given dataset: " + str(activities))
            exit(1)

        #Will likely match per subject in the future.
        #For now, include all activity samples together.
        datDataPathPattern = os.path.join(directory, "data", "{}_*.dat".format(activity))
        inputFiles = sorted(glob.glob(datDataPathPattern, recursive=True))
        
        inputWindows = []
        index = 0
        for inputFile in inputFiles:
            index += 1
            csiOutput = self.loadFromDat(inputFile, windowSize=windowSize, step=step)
            if csiOutput is not None:
                inputWindows.append(csiOutput)
            print("Loaded {:.2f}% for Activity: {}".format(index / len(inputFiles) * 100, activity))

        inputArray = np.concatenate(inputWindows, axis=0)

        if saveToFile:
            np.savez_compressed(compressedPath, inputArray)

        if downsample > 1:
            inputArray = inputArray[:, ::downsample, :]

        outputLabels = np.zeros((inputArray.shape[0], len(activities)))
        outputLabels[:, activities.index(activity)] = 1
        return inputArray, outputLabels

    def loadDataForActivities(self, directory, activities=ALL_ACTIVITIES, saveToFile=False, windowSize=FS, step=50, downsample=1):

        x_all = []
        y_all = []
        for activity in activities:
            inputArray, outputLabels = self.loadByActivityLabel(directory, activity, activities, saveToFile=saveToFile, windowSize=windowSize, step=step, downsample=downsample)
            x_all.append(inputArray)
            y_all.append(outputLabels)

        x_all = np.concatenate(x_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        return (x_all, y_all)

    def initialiseData(self, directory, downsample=1):

        self.x_all, self.y_all = self.loadDataForActivities(directory, saveToFile=True, downsample=downsample)