from experiment import Experiment

import argparse
import json
import os
import sys

VALID_DATASETS = ["csiar", "wiar", "csi46traces", "falldefi", "newintel", "newpi"]

def validateParams(args):
    config = args.config
    datasetCsv = args.datasets  
    resultsLoc = args.results

    # Verify config location is valid and then load it.
    if not os.path.exists(config):
        print("Config does not exist at location: " + config)
        exit(1)

    config = json.loads(open(config).read())

    #Verify supplied datasets are supported and valid.
    datasets = [x for x in datasetCsv.split(",") if x in VALID_DATASETS]
    
    if len(datasets) < 1:
        print("No valid datasets provided. Supported datasets are: csiar|wiar|csi46traces|falldefi")
        exit(1)

    #Verify no file currently exists at the given results location.
    # if os.path.exists(resultsLoc):
    #     print("File already exists at: " + resultsLoc)
    #     exit(1)
    
    resultsFile = open(resultsLoc, "w+")

    return config, datasets, resultsFile

def runExperiments(config, datasets, resultsFile):

    for dataset in datasets:
        if not config[dataset]:
            print("No config included for dataset: " + dataset)
            exit(1)

        datasetConfig = config[dataset]
        experiment = Experiment(dataset, config, resultsFile)


if __name__ == "__main__":

    # dataset types = csiar|wiar|csi46traces|falldefi
    parser = argparse.ArgumentParser(description="Run experiments given CSI datasets in a variety of formats.")
    parser.add_argument("-c", "--config", type=str, nargs="?", default="config.json", help="Specify a configuration file for experiments. (default: config.json)")
    parser.add_argument("-d", "--datasets", type=str, nargs="?", default="newpi", help="Specify which datasets to use for experiments. Use unspaced commas for more than one. (options: csiar|wiar|csi46traces|falldefi)")
    parser.add_argument("-r", "--results", type=str, nargs="?", default="results.json", help="Specify a location to save experimental results. (default: results.json")

    parsedArgs = parser.parse_args()

    config, datasets, resultsFile = validateParams(parsedArgs)
    runExperiments(config, datasets, resultsFile)