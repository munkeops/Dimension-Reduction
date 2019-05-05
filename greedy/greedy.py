import pandas
from sklearn.datasets import load_digits
digits = load_digits()

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
#print ("Images", digits.data.shape)
dataset=digits['data']
end = digits.target
#print ("Images data",dataset )

# Print to show there are 1797 labels (integers from 0–9)
#print("Label Data Shape", digits.target.shape)



import numpy as np 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:10], digits.target[0:10])):
    plt.subplot(1, 10, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 10) 
plt.show()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()
#train the regression model with original dataset 
logisticRegr.fit(x_train, y_train)


dimension=[0]*64
reduced=[0]*64


def classify(datas,end):
    #method to run the new data sets on the regression model to calculate accuracy
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(datas, end, test_size=0.25, random_state=0)
 
    # Returns a NumPy Array
    # Predict for One Observation (image)
    logisticRegr.predict(x_test[0].reshape(1,-1))

    predictions = logisticRegr.predict(x_test)
   # //print(predictions)

    # Use score method to get accuracy of model
    score = logisticRegr.score(x_test, y_test)
    return score*100
#/print(score*100)

def dimreduc(k,dataset,end):
    #function to reduce the dimenion of the data set one pixel at a time(one dimension at a time)
    datas = dataset.copy() 
    for i in range(0,1797):
        for j in range(0,64):
            if(j==k):
                datas[i][k]=0
    return classify(datas,end)

def dimreduc2(dataset,end,dims):
    #function to increase dimensions when re-building image (reduces original data set to the number of dimension wanted)
    datas = dataset.copy() 
    for i in range(0,1797):
        for j in range(0,64):
            if(j not in dims):
                datas[i][j]=0
    return classify(datas,end)
    
  
def dto2d(l):
    #fucntion to convert single row data into a 8x8 matrix
    onedarray = np.array(l)
    twodarray=onedarray.reshape(8,8)
    return twodarray

for i in range(0,64):
    #function to loop on the given data set to extract one pixel at a time and calculate accuracy without those pixels
    dimension[i] = dimreduc(i,dataset,end)


print(dimension)

matdimension=dto2d(dimension)


currentx=4
currenty=4
#print(current)
reduced=dto2d(reduced)
reduced[currentx][currenty]=matdimension[currentx][currenty]
l=[]
accuracy=0
dims = []
dims.append(8*currentx+currenty)

print(dims)
minval=95
while(accuracy<95 and minval <100):
    
    if((currentx-1>=0) and currenty-1>=0 ):
        l.append(matdimension[currentx-1][currenty-1])
    else :
        l.append(100) 
    if((currentx>=0) and currenty-1>=0):
        l.append(matdimension[currentx][currenty-1])
    else :
        l.append(100) 
    if((currentx>=0) and currenty-1>=0) and (currentx+1<=7):
        l.append(matdimension[currentx+1][currenty-1])
    else :
        l.append(100) 

    if((currentx-1>=0) and currenty>=0):
        l.append(matdimension[currentx-1][currenty])
    else :
        l.append(100)  
   
    if((currentx>=0) and currenty>=0 and (currentx+1<8) ):
        l.append(matdimension[currentx+1][currenty])
    else :
        l.append(100) 

    if((currentx-1>=0) and currenty>=0 and (currenty+1<8)):
        l.append(matdimension[currentx-1][currenty+1])
    else :
        l.append(100); 
    if((currentx>=0) and currenty>=0 and (currenty+1<8)):
        l.append(matdimension[currentx][currenty+1])
    else :
        l.append(100) 
    if((currentx>=0) and currenty>=0 and (currentx+1<8) and (currenty+1<8)):
        l.append(matdimension[currentx+1][currenty+1])
    else :
        l.append(100) 


    matdimension[currentx][currenty]=100
    minval=min(l)
    count=l.index(minval)
    l=[]
    if(count==0):
        currentx=currentx-1
        currenty=currenty-1
    if(count==1):
        currentx=currentx
        currenty=currenty-1
    if(count==2):
        currentx=currentx+1
        currenty=currenty-1
    if(count==3):
        currentx=currentx-1
        currenty=currenty
    if(count==4):
        currentx=currentx+1
        currenty=currenty
    if(count==5):
        currentx=currentx-1
        currenty=currenty+1
    if(count==6):
        currentx=currentx
        currenty=currenty+1
    if(count==7):
        currentx=currentx+1
        currenty=currenty+1
    reduced[currentx][currenty]=matdimension[currentx][currenty]
    #newlist = d2tod(reduced)
    dims.append(8*currentx+currenty)
    accuracy=dimreduc2(dataset,end,dims)
    print(accuracy)
print("hence by reducing the dimensions the only required dimenions to give approximately 91% accuracy through greeedy algorithm are mentioned below")
print(dims)
print(len(dims))

datas = dataset.copy() 
for i in range(0,1797):
    for j in range(0,64):
        if(j not in dims):
            datas[i][j]=0

import numpy as np 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:10], digits.target[0:10])):
    plt.subplot(2, 10, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Initial: %i\n' % label, fontsize = 10) 
for index, (image, label) in enumerate(zip(datas[0:10], end[0:10])):
    plt.subplot(2, 10, index + 1+10)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Final: %i\n' % label, fontsize = 10) 
plt.show()