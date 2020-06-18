import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import math
import time
import os
import cv2 as cv

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

hist_size = [255]
hist_range = [0,256]
dataset = []
path = r'C:\Users\rajsh\Desktop\GMM\Training Set\Green'

vid = cv.VideoCapture("detectbuoy.avi")
for i in os.listdir(path):
    dataset.append(i)

green_data = 28
orange_data = 111
yellow = 140

def GaussianEquation(sigma, x, mean):
    equation = (1/(sigma*math.sqrt(2*math.pi)))*np.exp(-0.5*(x-mean)**2/sigma**2)
    return equation

def AverageHistogram():
    mean_b = []
    mean_g = []
    mean_r = []
    
    std_dev_b = []
    std_dev_g = []
    std_dev_r = []
    histogram_r = np.zeros((255,1))
    histogram_g = np.zeros((255,1))
    histogram_b = np.zeros((255,1))

    # Iterate for no of green buoy images
    for i in range(0,green_data):
        string_path = path+"\green"+str(i)+".jpg"
        img = cv.imread(string_path) 
        color = ("b","g","r")

        # New Mean Calculation
        mask= np.zeros((img.shape[0],img.shape[0],3), np.uint8)
        coordinates = np.indices((img.shape[0], img.shape[0]))
        coordinates = coordinates.reshape(2, -1)
        x , y=coordinates[0] , coordinates[1]
        indices=np.where((x-img.shape[0]/2)**2+(y-img.shape[0]/2)**2 < (img.shape[0]/2)**2)
        xnew,ynew=x[indices[0]],y[indices[0]]
        mask[xnew,ynew]=img[xnew,ynew]
        #cv.imshow("e",mask)
        pixels=img[xnew,ynew]
        mean=np.sum(pixels, axis=0) / len(pixels)
        stds=[np.std(pixels[:,0]),np.std(pixels[:,1]),np.std(pixels[:,2])]

        
        for j,c in enumerate(color):
            if c == "b":
                temp_b = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_b = np.column_stack((histogram_b, temp_b))
                mean_b.append(mean[0])
                std_dev_b.append(stds[0])
            if c == "g":
                temp_g = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_g = np.column_stack((histogram_g, temp_g))
                mean_g.append(mean[1])
                std_dev_g.append(stds[1])
            if c == "r":
                temp_r = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_r = np.column_stack((histogram_r, temp_r))
                mean_r.append(mean[2])
                std_dev_r.append(stds[2])

    histogram_avg_b = np.sum(histogram_b, axis=1) / (green_data)
    histogram_avg_g = np.sum(histogram_g, axis=1) / (green_data)
    histogram_avg_r = np.sum(histogram_r, axis=1) / (green_data)

    #Uncomment to plot histograms
    #plt.subplot(3,1,1)
    #plt.plot(histogram_avg_b, color = "b")
    #plt.subplot(3,1,2)
    #plt.plot(histogram_avg_g, color = "g")
    #plt.subplot(3,1,3)
    #plt.plot(histogram_avg_r, color = "r")
    #plt.show()
    
    return mean_r, mean_g, mean_b, std_dev_r, std_dev_g, std_dev_b

def EM():
    K = 5
    datapoint_b = []
    datapoint_g = []
    datapoint_r = []
    
    for i in range(0, green_data):
        string_path = path+"\green"+str(i)+".jpg"
        img = cv.imread(string_path)
        blue_chan_img = img[:,:,0]
        green_chan_img = img[:,:,1]
        red_chan_img = img[:,:,2]
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[0]):
                if (i-img.shape[0]/2)**2+(j-img.shape[0]/2)**2 < (img.shape[0]/2)**2:
                    datapoint_b.append(blue_chan_img[i,j])
                    datapoint_g.append(green_chan_img[i,j])
                    datapoint_r.append(red_chan_img[i,j])
  
    #intital estimates
    mean_b_init = 120
    mean_g_init = 240
    mean_r_init = 125
    std_dev_b_init = 40
    std_dev_g_init = 40
    std_dev_r_init = 40
    
    iterations = 0
    
    while(iterations <=70):        
        responsibility_1 = []
        responsibility_2 = []
        responsibility_3 = []
        
        probability_dist_1 = []
        probability_dist_2 = []
        probability_dist_3 = []
        
        pi_k_1 = []
        pi_k_2 = []
        pi_k_3 = []            
        
        #perform e and m for each datapoint
        for i in range(len(datapoint_b)):
            #calculate probabilty at that pixel
            probability_1 = GaussianEquation(std_dev_b_init, datapoint_b[i], mean_b_init)
            probability_2 = GaussianEquation(std_dev_g_init, datapoint_g[i], mean_g_init)
            probability_3 = GaussianEquation(std_dev_r_init, datapoint_r[i], mean_r_init)
            
            #gaussian of 3 channels
            probability_dist_1.append(probability_1)
            probability_dist_2.append(probability_2)
            probability_dist_3.append(probability_3)
            
            temp_pi_1 = probability_1/(probability_1 + probability_2 + probability_3)
            temp_pi_2 = probability_2/(probability_1 + probability_2 + probability_3)
            temp_pi_3 = probability_3/(probability_1 + probability_2 + probability_3)
            
            pi_k_1.append(temp_pi_1)
            pi_k_2.append(temp_pi_2)
            pi_k_3.append(temp_pi_3)

        #formula for calculating new mean from pdf    
        mean_b_init = np.sum(np.array(pi_k_1)*np.array(datapoint_b))/np.sum(np.array(pi_k_1))
        mean_g_init = np.sum(np.array(pi_k_2)*np.array(datapoint_g))/np.sum(np.array(pi_k_2))
        mean_r_init = np.sum(np.array(pi_k_3)*np.array(datapoint_r))/np.sum(np.array(pi_k_3))
        
        #calculating SD from mean and data points
        std_dev_b_init = (np.sum(np.array(pi_k_1) * ((np.array(datapoint_b)) 
        - mean_b_init) ** (2)) / np.sum(np.array(pi_k_1))) ** (1 / 2)
        std_dev_g_init = (np.sum(np.array(pi_k_2) * ((np.array(datapoint_g)) 
        - mean_g_init) ** (2)) / np.sum(np.array(pi_k_2))) ** (1 / 2)
        std_dev_r_init = (np.sum(np.array(pi_k_1) * ((np.array(datapoint_r)) 
        - mean_r_init) ** (2)) / np.sum(np.array(pi_k_3))) ** (1 / 2)      
        
        iterations = iterations + 1
        print(iterations)
    return mean_b_init, mean_g_init, mean_r_init, std_dev_b_init, std_dev_g_init, std_dev_r_init  




if __name__ == "__main__":
    mean_r, mean_g, mean_b, std_dev_r, std_dev_g, std_dev_b = AverageHistogram()
    avg_mean_b = sum(mean_b)/len(mean_b)
    avg_mean_g = sum(mean_g)/len(mean_g)
    avg_mean_r = sum(mean_r)/len(mean_r)
    
    avg_std_dev_b = sum(std_dev_b)/len(std_dev_b)
    avg_std_dev_g = sum(std_dev_g)/len(std_dev_g)
    avg_std_dev_r = sum(std_dev_r)/len(std_dev_r)
    
    gaussian_b = GaussianEquation(avg_std_dev_b, list(range(0,256)), avg_mean_b)
    gaussian_g = GaussianEquation(avg_std_dev_g, list(range(0,256)), avg_mean_g)
    gaussian_r = GaussianEquation(avg_std_dev_r, list(range(0,256)), avg_mean_r)
    
    mean_b_init, mean_g_init, mean_r_init, std_dev_b_init, std_dev_g_init, std_dev_r_init = EM()
    
    
    greenboi_r = GaussianEquation(std_dev_r_init, list(range(0,256)), mean_r_init)
    greenboi_g = GaussianEquation(std_dev_g_init, list(range(0,256)), mean_g_init)
    greenboi_b = GaussianEquation(std_dev_b_init, list(range(0,256)), mean_b_init)
    plt.plot(greenboi_r, "r", greenboi_g, "g", greenboi_b, "b")
    #plt.show()
    
    while True:
      ret,frame = vid.read()
      if frame is not None:
        frame_b=frame[:,:,0]
        frame_g=frame[:,:,1]
        frame_r=frame[:,:,2]
        if ret == True:
            frame_updated=np.zeros(frame_g.shape, dtype = np.uint8)
            coordinates = np.indices((frame_g.shape[0], frame_g.shape[1]))
            coordinates = coordinates.reshape(2, -1)
            x,y=coordinates[0],coordinates[1]
            pixel_valr=frame_r[x,y]
            pixel_valg=frame_g[x,y]
            pixel_valb=frame_b[x,y]
            indices1=np.where((greenboi_g[pixel_valg]>0.04))
            #indices1=np.where((greenboi_r[pixel_valr]>0.001) & (greenboi_g[pixel_valg]>0.005) & (greenboi_b[pixel_valb]>0.004) )
            x1,y1=x[indices1[0]],y[indices1[0]]
            frame_updated[x1,y1]=255
            
            kernel_square = np.ones((10,10),np.uint8)
            kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
            np.array([[0, 0, 1, 0, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 0, 1, 0, 0]], dtype=np.uint8)
            blur = cv.blur(frame_updated,(10,10)) 
            ret_thresh, thresholded = cv.threshold(blur, 50, 255, cv.THRESH_BINARY)
            edges = cv.Canny(thresholded, 200, 300)
            dilated = cv.dilate(thresholded, kernel_ellipse, iterations = 1)
            _,contours, _ = cv.findContours(dilated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            #cont_img = cv.drawContours(frame, contours, -1, (0,0,255), 5)

            # Draw circle to fit the contours enclosing specified area
            for c in contours:
                if cv.contourArea(c) > 40:
                    print("inside1")
                    (x,y),r = cv.minEnclosingCircle(c)
                    center = (int(x),int(y))
                    r = int(r)
                    print(r)
                    if r > 8 and r < 15 and y<350 and y>150 :
                        print("inside")
                        cv.circle(frame,center,r,(0,255,0),2)
            cv.imshow("threshold", frame)
            k = cv.waitKey(15) & 0xff
            if k == 27:
                break

        else:
            break
         
    vid.release()
    cv.destroyAllWindows()
