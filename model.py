import tensorflow as tf

class Model(object):

    def __init__(self, config, structure="deepconvlstm"):
        self.structure = structure
        
        if structure == "deepconvlstm":
            self.model = self.buildDeepConvLSTM(config)
    
    def buildDeepConvLSTM(self, config, visualiseModel="model_image.png"):

        inputWindowLength = config["inputWindowLength"]
        inputValuesLength = config["inputValuesLength"]

        outputValuesLength = config["outputValuesLength"]
        outputActivation = config["outputActivation"]

        convLayers = config["convolutionalLayers"]

        convFilters = config["convolutionalFilters"]
        convKernelSize = config["convolutionalKernelSize"]
        convActivation = config["convolutionalActivationFunction"]
        convStrides = config["convolutionalStrides"]

        poolingPoolSize = config["poolingPoolSize"]

        lstmLayers = config["lstmLayers"]
        lstmUnits = config["lstmUnits"]

        model = tf.keras.Sequential()

        #Input layer.
        #This should be a 2D vector:
        #   x = window length (number of timesteps)
        #   y = input values (number of features)
        model.add(tf.keras.Input(shape=(inputWindowLength, inputValuesLength)))

        #Convolutional layers.
        #These are 1D convolutional layers since we are training on time series data.
        for x in range(convLayers):
            model.add(tf.keras.layers.Conv1D(convFilters, convKernelSize, padding="valid", activation=convActivation, strides=convStrides))

        #Pooling layer.
        #Scope to add more as necessary.
        #May also want to layer them in with the convolutional layers.
        model.add(tf.keras.layers.MaxPooling1D(pool_size=poolingPoolSize))
        
        for x in range(lstmLayers):
            return_sequences = True if x != lstmLayers-1 else False
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstmUnits, return_sequences=return_sequences)))

        model.add(tf.keras.layers.Dense(outputValuesLength, activation=outputActivation))
        
        # if len(visualiseModel) > 0:
        #     tf.keras.utils.plot_model(model, to_file=visualiseModel, show_shapes=True, show_layer_names=True)
        #     print("Model image written to: " + visualiseModel)

        return model