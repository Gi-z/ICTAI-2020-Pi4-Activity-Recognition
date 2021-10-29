import os

class Handler(object):

    def __init__(self, directory, config):

        self.config = config

        if not os.path.exists(directory):
            print("Directory for dataset does not exist at: " + directory)
            exit(1)
        
        self.path = directory

    def initialiseData(self):
        print("Use a subclassed handler method for specific datasets.")
        exit(1)