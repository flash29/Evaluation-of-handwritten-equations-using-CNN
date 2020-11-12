# Evaluation of handwritten equations using CNN

Evaluation of Handwritten Expression using CNN(Convolutional Neural Networks)

The CNN is trained with the images in the datasetwhich has the 13 classes (0-9,+,-,*)(cnn.py)

The input image given from the user is segmented and using our trained CNN we predict the class of the images 
which is stored in the form of string and we perform string operations to get the result.(imagesegment.py)


To run the program :

1. Download the files.

2. upload the image you want to test into the folder and update the name of the image in the path of the image in the segment.py file.

3. Then run the program by using an ide like spyder 

4. We will get the the final result of the equation, the segmented images of individual charecters and the countours on the main image. 
