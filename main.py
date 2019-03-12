import numpy as np
import cv2
import time
import math

"""
Created on Wed Oct 10 16:07:33 2018

@author: Dikhsheet Raya
"""
def dist(pt1, pt2):
    x1, _ = pt1
    x2, _ = pt2
    x = x1 - x2
    print(x)
    if (abs(x) <= 20):
        return 0
    else:
        return abs(x)/x

def count_white(img):
    '''count white pixels'''
    output=list()
    for i in range(img.shape[1]):
        count=0
        for j in range(img.shape[0]):
            if(img[j, i] > 200):
                count += 1
        output.append(count)
    return output

def find_centres_2(histo):
    '''finds the centwrs based ono average'''
    sum=0;
    count=1;
    #std=0;
    for i in range(len(histo)):
        if(histo[i]>3):
            sum+=i
            count+=1
    mean=sum/count;
    #for i in range(len(out1)):
        #std+=((i-avg)**2)
    #std=std/count
    #print("std ", std**(1/2))
    #return std,avg
    return mean

def find_centres_3(out1):
    sum=0;
    count=1;
    for i in range(len(out1)):
        sum+=out1[i]*i
        count+=out1[i]
    return sum/count

def process(img):

    img = cv2.GaussianBlur(img,(5,5), 101)
    _,img=cv2.threshold(img,80,255,cv2.THRESH_BINARY)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j]= 255 - img[i,j]

    rows = img.shape[0]
    n = 5
    kernel = np.ones((9,9), np.uint8)

    img = cv2.dilate(img, kernel, iterations=1)
    win_size = int(rows / n)
    result=list()
    #std_out= list()
    slices = range(0, rows, win_size)
    size= len(slices) - 1
    for i in range(size):
        temp = img[slices[i] : slices[i+1],:]
        histo = count_white(temp)
        centre = find_centres_2(histo)#std,

        #std_out.append(std)
        result.append([(slices[i] + slices[i + 1]) / 2, centre])
    result=np.asarray(result,dtype=int)
    return result#,std_out

cap = cv2.VideoCapture("MOV_0112.mp4")



while(1):
    print("hello")
    ret, frame = cap.read()
    frame = cv2.pyrDown(frame,(120,200))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output=process(gray)
    #print(output,"\nstd= ",std)
    ac_centre=output[:,0]
    print (ac_centre)
    print(ac_centre.shape)
    print(ac_centre)
    pt_centre=[]
    #for i in range(len(output)):
    p1=(int(output[-2,1]),int(output[-2,0]))
    p0=(int(frame.shape[-2]/2),int(ac_centre[-2]))
    frame=cv2.circle(frame,(int(output[-2,1]),int(output[-2,0])),6,(0,0,255),-1)
    frame=cv2.line(frame,p0,p1,(0,255,255))
    setter = dist(p0,p1)
        #txt=str(std[2])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,str(setter),(100,100),font,1,(0,255,0),2)

    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for i in range(5):
        ret,frame=cap.read()

cap.release()
cv2.destroyAllWindows()
