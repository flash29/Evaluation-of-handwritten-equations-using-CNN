

from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
from matplotlib import pyplot as plt

####loading the model
json_file = open('model_final3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_final3.h5")

# import image

image = cv2.imread('-_289.jpg')



# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)   
plt.show()        
#cv2.imshow('gray', gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# binary
ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
plt.imshow(thresh)   
plt.show() 
#cv2.imshow('threshold', thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# dilation
kernel = np.ones((10, 1), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
plt.imshow(img_dilation)   
plt.show() 
#cv2.imshow('dilated', img_dilation)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# find contours
# cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
cv2MajorVersion = cv2.__version__.split(".")[0]
# check for contours on thresh
if int(cv2MajorVersion) >= 4:
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
else:
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
train_data=[]
for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)    
    im_crop =img_dilation[y:y+h+10,x:x+w+10]
    im_resize = cv2.resize(im_crop,(200,200))
    plt.imshow(im_resize)   
    plt.show() 
#    cv2.imshow("work",im_resize)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    im_resize=np.reshape(im_resize,(200,200,1))
    train_data.append(im_resize)


plt.imshow(image)   
plt.show() 
#cv2.imshow('marked areas', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


####prediction

r=''
for i in range(len(train_data)):
    train_data[i]=np.array(train_data[i])
    train_data[i]=train_data[i].reshape(1,200,200,1)
    result=loaded_model.predict_classes(train_data[i])
    print(result)
    if(result[0]==0):
        r=r+'-'
    if(result[0]==1):
        r=r+'+'
    if(result[0]==2):
        r=r+'0'
    if(result[0]==3):
        r=r+'1'
    if(result[0]==4):
        r=r+'2'
    if(result[0]==5):
        r=r+'3'
    if(result[0]==6):
        r=r+'4'
    if(result[0]==7):
        r=r+'5'
    if(result[0]==8):
        r=r+'6'
    if(result[0]==9):
        r=r+'7'
    if(result[0]==10):
        r=r+'8'
    if(result[0]==11):
        r=r+'9'
    if(result[0]==12):
        r=r+'*'
    
print(r)   
eval(r)
