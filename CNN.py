
 # Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
 
# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution
classifier.add(Convolution2D(32,( 3, 3), input_shape = (200,200,1), activation = 'relu'))
 
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Adding a second convolutional layer
classifier.add(Convolution2D(16, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(13, activation = 'softmax'))
 
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
 
 
from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   )
 
test_datagen = ImageDataGenerator(rescale = 1./255)

cl=['-','+','0','1','2','3','4','5','6','7','8','9','times']
 
training_set = train_datagen.flow_from_directory('dataset/dataset/train',
                                                 target_size = (200,200),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
                                                 color_mode="grayscale",
                                                 classes=cl
                                                 )
 
test_set = test_datagen.flow_from_directory('dataset/dataset/test',
                                            target_size = (200,200),
                                            batch_size = 32,
                                            class_mode = 'categorical',
                                            color_mode="grayscale",
                                            classes=cl
                                            )

#imgs , labels = next(training_set)


 
classifier.fit_generator(training_set,
                         epochs = 10,
                         validation_data = test_set,
                         
                        )

#classifier.fit(training_set,epochs=10,validation_data=test_set)


#to save the model 
model_json = classifier.to_json()
with open("model_final3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to
classifier.save_weights("model_final3.h5")


