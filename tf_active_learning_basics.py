# minimum_evaluation_items = 1200 # annotate this many randomly sampled items first for evaluation data before creating training data
# minimum_training_items = 400 # minimum number of training items before we first train a model
#
# epochs = 10 # number of epochs per training session
# select_per_epoch = 200  # number to select per epoch per label
#
#
# data = []
# test_data = []


import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_labels, vocab_size):
        super(MyModel, self).__init__()
        # Is there an assumption that there is an activation in there
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(num_labels)

    def call(self, inputs):
        x = self.dense1(inputs) #ReLU activation already part of the layer
        return self.dense2(x)

def train_model(training_data, validation_data = "", evaluation_data = "", num_labels=2, vocab_size=0):

    # Negative Log Likelihood loss is what is being used within
    # TODO - Still working on the loss function
    loss_object = tf.keras.losses.??

    # What exactly is it discussing when it is referencing model parameters
    # TODO - This is assuming that all of the other parameters are the same as in pytorch
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
