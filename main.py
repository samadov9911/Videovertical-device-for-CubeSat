import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.optimize import minimize
from scipy.optimize import Bounds
import math

#Read image from
img = cv2.imread('images/45.jpg')


#Add filter to image for blur
blur_image = cv2.GaussianBlur(img,(3, 33), 0)


#Conver to gray
gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)


#make white and black pixels
_, thres = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

#Highlight edges
edged = cv2.Canny(thres, 10, 255)


#just size of image
height = len(gray[:,1])
width = len(gray[1,:])


#Array of white pixels with coords
indices = np.where(edged != 0)
X = [elem for elem in indices[1] ]
Y = [elem for elem in indices[0] ]



#make array with center
if len(X) % 2 == 0:
    X.append(min(X))
    Y.append(min(Y))




LeftX =  min(X) 
RightX = max(X) 
CenterX= np.median(X) 

indec = np.where(X == CenterX)[0][0]

LeftY = Y[np.argmin(X)] 
RightY = Y[np.argmax(X)]
CenterY =  Y[indec] 



X=[LeftX,CenterX,RightX]
Y=[LeftY,CenterY,RightY]



plt.imshow(img)
plt.plot(X,Y)

plt.show()



def error(args):
    sum=0
    for i, elem in enumerate(X):
        sum=sum+((args[0]-X[i])**2+(args[1]-Y[i])**2-args[2]**2)**2
    return sum

result = minimize(error, np.array([0, 0, 5000]), 
                  method = 'L-BFGS-B', 
                  bounds=((-10000,10000),(-10000,10000),(-10000,10000)))


coef = result.x

print(coef)

xc = coef[0] - round(width/2)

yc = coef[1] - round(height/2)

print(xc ,yc)


theta = -math.atan2(xc, yc) * 180/math.pi

print(theta)




