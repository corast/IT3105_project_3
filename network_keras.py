from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import backend as K
from keras.optimizers import SGD,Adam,Adagrad,RMSprop
import Datamanager 

class Model():
    def __init__(self,model:Sequential, optimizer, loss):
        self.model = Sequential(model) # model should be an keras model
        print(model)
        self.optimizer = optimizer
        self.loss = loss
        #self.input_shape= (height, width, depth)
        #if(K.image_data_format()=="channels_first"):
        #    self.input_shape = (depth, height, width)


    def store(self):
        pass

    def train(self, datamanager:Datamanager.Datamanager, epochs, batch_size):
        datamanager.return_keras()# Return all data in CSV file. 
        #self.model.fit(epochs=epochs,batch_size=batch_size)
        #self.model.train_on_batch(batch_size)
        #pass

model_1 = Model(model=[
    Conv2D(data_format="channel_first",filters=5,input_shape=(3,5,5),kernel_size=(3,3),padding=1,activation="relu"),
    Flatten(),
    Dense(25, activation="softmax")], 
    optimizer="adam", loss="categorical_crossentropy")

sgd = SGD
"""
model = Sequential()
model.add(Conv2D(3,kernel_size=3,activation="relu",input_shape=(3,5,5)))
model.add(Flatten())
model.add(Dense(25,activation="softmax"))

dataset_train = Datamanager.Datamanager("Data/random_15000.csv",dim=5, modus=2)

model.compile(optimizer="adam",loss="categorical_crossentropy")
model.fit()
"""