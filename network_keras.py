from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import backend as K
from keras.optimizers import SGD,Adam,Adagrad,RMSprop
import Datamanager 
import os
class Model():
    def __init__(self,model:Sequential, optimizer, loss, name, filepath=None, input_type=1):
        self.model = Sequential(model) # model should be an keras model
        print(model)
        self.optimizer = optimizer
        self.loss = loss
        self.name = name
        self.input_type = input_type

        if(filepath is not None):
            self.load(filepath)

        #self.input_shape= (height, width, depth)
        #if(K.image_data_format()=="channels_first"):
        #    self.input_shape = (depth, height, width)


    def store(self,filepath,epoch):
        # Store model at specific filepath.
        save_dir = "models/"+self.name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_path = save_dir+"/"+self.name+".json"
        if not os.path.exists(model_path):
            with open(model_path,"w") as json_file:
                json_file.write(model_path)
        weight_path = save_dir+"/"+ self.name + "_" + str(epoch)+".h5" # save a new network with an unique ID, name + epoch
        self.model.save_weights(weight_path)

    def load(self, filepath):
        if(os.path.isfile(filepath)):
            # Dont need to open json file.
            self.model.load_weights(filepath)
            self.model.compile(loss=self.loss,optimizer=self.optimizer)

    def train(self, datamanager:Datamanager.Datamanager, iterations, batch_size):
        datamanager.return_keras()# Return all data in CSV file. 
        self.model.fit(epochs=iterations,batch_size=batch_size)
        #self.model.train_on_batch(batch_size)
        #pass
    
    def evaluate(self, datamanager:Datamanager.Datamanager, batch_size):
        datamanager.return_batch_keras()
        self.model.evaluate()

model_1 = Model(model=[
    Conv2D(data_format="channel_first",filters=5,input_shape=(3,5,5),kernel_size=(3,3),padding=1,activation="relu"),
    Flatten(),
    Dense(25, activation="softmax")], 
    optimizer="adam", loss="categorical_crossentropy",name="model-test")

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