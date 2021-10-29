from .handler import Handler

import glob
import os

import numpy as np
import pandas as pd

class CSIAR(Handler):

    FS = 1000
    ALL_ACTIVITIES = ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]

    def __init__(self, directory, config):
        super().__init__(directory, config)

        self.initialiseData(directory, config["downsample"])

    def loadFromCSV(self, inputFile, labelFile, windowSize=FS, threshold=0.6, step=200):
        annotationsPd = pd.read_csv(labelFile, names=["label"])
        annotationsPd = pd.get_dummies(annotationsPd.label).NoActivity
        annotations = 1-np.array(annotationsPd.values)

        features = []
        inputPd = pd.read_csv(inputFile, header=None).values
        for line in inputPd:
            lineData = np.array([float(x) for x in line])

            #Throwing out timestamp (0), and phase data (>91)
            # lineData = lineData[1:91]
            lineData = lineData[1:31]
            lineData = lineData - np.mean(lineData)

            #As lineData is a numpy array, it needs to be double wrapped
            #so the complete features list can be concatenated along the first axis.
            #The same could be achieved with reshape((1, x, y))
            features.append(lineData[np.newaxis,...])

        features = np.concatenate(features, axis=0)
        assert(features.shape[0] == annotations.shape[0])

        #Data needs to be split into windows.
        #   We're going to move over 1000 sample sections of the data,
        #   with a sliding step of 200 samples. If there are more than
        #   600 positive activity samples in a window, that window
        #   will be stored. Otherwise, move onto the next step.
        #   
        #   All windows stored in mergedInput are considered positive
        #   windows for the current activity.

        index = 0
        positiveInput = []
        while index + windowSize <= features.shape[0]:
            curActivity = annotations[index:index+windowSize]
            sumActivity = np.sum(curActivity)
            if sumActivity < threshold * windowSize:
                index += step
                continue
            # curFeature = np.zeros((1, windowSize, 90))
            curFeature = np.zeros((1, windowSize, 30))
            curFeature[0] = features[index:index+windowSize, :]
            positiveInput.append(curFeature)
            index += step

        return np.concatenate(positiveInput, axis=0)

    def loadByActivityLabel(self, directory, activity, activities=ALL_ACTIVITIES, saveToFile=False, windowSize=FS, threshold=0.6, step=200, downsample=1):

        compressedFilename = "x_{}_win_{}_threshold_{}percent_step_{}.npz".format(activity, windowSize, int(threshold*100), step)
        compressedPath = os.path.join(directory, compressedFilename)

        if os.path.exists(compressedPath):
            print("Compressed data for activity with experiment parameters exists.")
            print("Loading: " + compressedPath)

            inputArray = np.load(compressedPath)["arr_0"]

            if downsample > 1:
                inputArray = inputArray[:, ::downsample, :]

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
        csvDataPathPattern = os.path.join(directory, "Data", "input_*" + activity + "*.csv")
        inputFiles = sorted(glob.glob(csvDataPathPattern))
        annotationFiles = [os.path.basename(filename).replace("input_", "annotation_") for filename in inputFiles]
        annotationFiles = [os.path.join(directory, "Data", filename) for filename in annotationFiles]

        inputWindows = []
        index = 0
        for inputFile, annotationFile in zip(inputFiles, annotationFiles):
            index += 1
            if not os.path.exists(annotationFile):
                print("Annotation file could not be found at: " + annotationFile)
                continue
            inputWindows.append(self.loadFromCSV(inputFile, annotationFile, windowSize=windowSize, threshold=threshold, step=step))
            print("Loaded {:.2f}% for Activity: {}".format(index / len(inputFiles) * 100, activity))

        inputArray = np.concatenate(inputWindows, axis=0)
        if saveToFile:
            np.savez_compressed(compressedPath, inputArray)

        if downsample > 1:
            inputArray = inputArray[:, ::downsample, :]

        outputLabels = np.zeros((inputArray.shape[0], len(activities)))
        outputLabels[:, activities.index(activity)] = 1
        return inputArray, outputLabels

    def loadDataForActivities(self, directory, activities=ALL_ACTIVITIES, saveToFile=False, windowSize=FS, threshold=0.6, step=200, downsample=1):

        x_all = []
        y_all = []
        for activity in activities:
            inputArray, outputLabels = self.loadByActivityLabel(directory, activity, activities, saveToFile=saveToFile, windowSize=windowSize, threshold=threshold, step=step, downsample=downsample)
            x_all.append(inputArray)
            y_all.append(outputLabels)

        x_all = np.concatenate(x_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        return (x_all, y_all)

    def initialiseData(self, directory, downsample=1):

        #   CSIAR data is structured as:
        #       - Data
        #           - annotation_ACTIVITY_SUBJECT_SAMPLE.csv
        #           - input_ACTIVITY_SUBJECT_SAMPLE.csv
        #
        #   This source data is captured at 1000Hz.
        #   Activities:
        #       - Bed
        #       - Fall
        #       - Pickup
        #       - Run
        #       - Sitdown
        #       - Standup
        #       - Walk
        
        self.x_all, self.y_all = self.loadDataForActivities(directory, saveToFile=True, downsample=downsample)


if __name__ == "__main__":
    print("Running CSIAR tests.")

    directory = "E:\\Datasets\\CSIAR"
    config = {}

    test = CSIAR(directory, config)

    # testInput = "E:\\Datasets\\CSIAR\\Data\\input_bed_170308_1405_01.csv"
    # testAnnotation = "E:\\Datasets\\CSIAR\\Data\\annotation_bed_170308_1405_01.csv"

    # mergedInput = test.loadFromCSV(testInput, testAnnotation)
    # x = True

    x_all, y_all = test.loadDataForActivities(directory, test.ALL_ACTIVITIES, True)