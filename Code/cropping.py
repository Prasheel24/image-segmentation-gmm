import cv2 as cv
import numpy as np
import math
from random import randint


#https://www.life2coding.com/crop-image-using-mouse-click-movement-python/
#https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
v = []
y=0
o=0
g=0
yt=0
ot=0
gt=0
count=1
r=[]
#image = cv.imread('frame2.jpg') 

path_y = r'C:\Users\rajsh\Desktop\GMM\Training Set\Yellow'
path_o = r'C:\Users\rajsh\Desktop\GMM\Training Set\Orange'
path_g = r'C:\Users\rajsh\Desktop\GMM\Training Set\Green'
path_yt = r'C:\Users\rajsh\Desktop\GMM\Test\Yellow'
path_ot = r'C:\Users\rajsh\Desktop\GMM\Test\Orange'
path_gt = r'C:\Users\rajsh\Desktop\GMM\Test\Green'

def crop(p,q,l,crop,c,count):
    global y,o,g,yt,gt,ot
    crop=crop[q-l:q+l,p-l:p+l]
    length=2*l
    h,w,_=crop.shape
    mask1 = np.zeros((h,w), np.uint8)
    cv.circle(mask1,(l,l),l,(255,255,255),thickness=-1)
    crop = cv.bitwise_and(crop,crop,mask=mask1)
    crop[np.where((crop==[0,0,0]).all(axis=2))] = [255,255,255];
    if count<141:
        if c==1:
            path="\yellow"+str(y)+".jpg"
            cv.imwrite(path_y+path,crop)
            y=y+1
        if c==2:
            path="\orange"+str(o)+".jpg"
            cv.imwrite(path_o+path,crop)
            o=o+1
        if c==3:
            path="\green"+str(g)+".jpg"
            cv.imwrite(path_g+path,crop)
            g=g+1
    else:
        if c==1:
            path="\yellow"+str(yt)+".jpg"
            cv.imwrite(path_yt+path,crop)
            yt=yt+1
        if c==2:
            path="\orange"+str(ot)+".jpg"
            cv.imwrite(path_ot+path,crop)
            ot=ot+1
        if c==3:
            path="\green"+str(gt)+".jpg"
            cv.imwrite(path_gt+path,crop)
            gt=gt+1
        

def mouse_crop(event, x, y, flag, params):
    global count
    yellow=0
    green=0
    orange=0
    color = (0, 0, 0)
    if event==cv.EVENT_LBUTTONDOWN:
        v.append((x,y))
        if len(v)==2:
          l=math.floor(math.sqrt((v[1][0] - v[0][0])**2 + (v[1][1] - v[0][1])**2))
          crop(v[0][0],v[0][1],l,params,1,count)
          image=cv.circle(params, (v[0][0],v[0][1]), math.ceil(math.sqrt((v[1][0] - v[0][0])**2 + (v[1][1] - v[0][1])**2)), color, 1)
          cv.imshow("name",params)
        if len(v)==4:
          l=math.ceil(math.sqrt((v[3][0] - v[2][0])**2 + (v[3][1] - v[2][1])**2))
          crop(v[2][0],v[2][1],l,params,2,count)
          image=cv.circle(params, (v[2][0],v[2][1]), math.ceil(math.sqrt((v[3][0] - v[2][0])**2 + (v[3][1] - v[2][1])**2)), color, 1)
          cv.imshow("name",params)
        if len(v)==6:
          l=math.ceil(math.sqrt((v[5][0] - v[4][0])**2 + (v[5][1] - v[4][1])**2))
          crop(v[4][0],v[4][1],l,params,3,count)
          image=cv.circle(params, (v[4][0],v[4][1]), math.ceil(math.sqrt((v[5][0] - v[4][0])**2 + (v[5][1] - v[4][1])**2)), color, 1)
          cv.imshow("name",params)

   
    

r=[]
print(count)
while count<200:
  value = randint(0, 199)
  if value not in r:
    r.append(value)
    string="frame"+str(value)+".jpg"
    print(string)
    image=cv.imread(string)
    instruction1="To Prepare Dataset:"
    instruction2="Select the center and then border of the buoy"
    instruction3="Order of selecting buoy's is Yellow, Orange, Green"
    instruction4="Press Space key to change frame"
    image=cv.putText(image,instruction1,(2,20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    image=cv.putText(image,instruction2,(2,40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    image=cv.putText(image,instruction3,(2,60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    image=cv.putText(image,instruction4,(2,80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    cv.imshow("name",image)
    cv.setMouseCallback("name",mouse_crop,image)
    cv.waitKey(0)
    v.clear()
    count=count+1
    print(count)

cv.waitKey(0)
