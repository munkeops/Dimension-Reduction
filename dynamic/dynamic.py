

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
#plt.show()


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

tempo =[0]
for i in range(0,64):
    #function to loop on the given data set to extract one pixel at a time and calculate accuracy without those pixels
    tempo[0]=i
    dimension[i] = dimreduc2(dataset,end,tempo)


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
"""
print(dims)
minval=95
maxval=5
counter=0
oldaccuracy=0
while( maxval >0 and counter<63):
    counter=counter+1
    
    matdimension[currentx][currenty]=0

    for i in range(0,8):
        for j in range(0,8):
            if(matdimension[i][j]==0):
                pass
            else:
                l.append([i,j,matdimension[i][j]])
        

    m=[]
    for i in l:
        m.append(i[2])
    maxval=max(m)
    count=m.index(maxval)
    currentx=l[count][0]
    currenty=l[count][1]
    l=[]
    
    reduced[currentx][currenty]=matdimension[currentx][currenty]
    #newlist = d2tod(reduced)
    if((8*currentx+currenty) not in dims):
        dims.append(8*currentx+currenty)
   
    accuracy=dimreduc2(dataset,end,dims)
    if(accuracy <=oldaccuracy):
        dims.pop()
    oldaccuracy = accuracy    
    print(maxval)
    print(accuracy)
    print(currentx ," ", currenty)

print("hence by reducing the dimensions the only required dimenions to give approximately ",accuracy," accuracy through greeedy algorithm are mentioned below")
print(dims)
print(len(dims))

"""
oldmax=0
final=[]
l=[]
k=[]   #points we want
main=[]  #all values of single points

for i in range(0,64):
    k.append(i)
    l.append(dimreduc2(dataset,end,k))
    k=[]
main.append(l)
l=[]
pix=0
print(main)
maxindex=[]
while(max(main[pix])<95):
    k.append(main[pix].index(max(main[pix])))
    for i in range (0,64):
        #print('hi')
        if(i not in maxindex):      
        #print('bye')
            
            k.append(i)
            l.append(dimreduc2(dataset,end,k))
            k.pop()
            #print(k)

        else:
            l.append(0)
    #print(main[pix])
    maxval=max(main[pix])
    if(oldmax>=maxval):
        pass
    else:
        if(main[pix].index(max(main[pix])) not in final):
            final.append(main[pix].index(max(main[pix]))) 
        

    print("max",maxval)
    print(main[pix].index(max(main[pix])))
    if(main[pix].index(max(main[pix])) not in maxindex):
        maxindex.append(main[pix].index(max(main[pix])))     
    main.append(l)
    l=[]
    pix=pix+1
    oldmax=maxval


print(final)
print(len(final))

datas = dataset.copy() 
for i in range(0,1797):
    for j in range(0,64):
        if(j not in final):
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
