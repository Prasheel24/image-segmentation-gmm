# https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95
# https://towardsdatascience.com/gaussian-mixture-modelling-gmm-833c88587c7f
# https://github.com/DFoly/Gaussian-Mixture-Modelling/blob/master/gaussian-mixture-model.ipynb
# https://cmsc426.github.io/colorseg/#colorclassification

from matplotlib import pyplot as plt
import numpy as np
import math
import time
import os
import sys
try:
    # print("Ye")
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2 as cv
from scipy.stats import multivariate_normal as mvn
from imutils import contours

print("Enter a number based on the following options: ")
print("1 -> Histogram check to get clusters")
print("2 -> Train the model and check the log likelihood graph")
print("3 -> Detect buoy directly with saved parameters")
val = input("Your option: ") 

path_yellow = '/home/prasheel/Workspace/ENPM673/Project3/buoy-detection/Training Set/Yellow'
path_orange = '/home/prasheel/Workspace/ENPM673/Project3/buoy-detection/Training Set/Orange'
path_green = '/home/prasheel/Workspace/ENPM673/Project3/buoy-detection/Training Set/Green'

vid = cv.VideoCapture("detectbuoy.avi")
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
if val == "3":
    flag = input("Press 1 to save the output to video else 0: ")
    if flag == "1":
        out = cv.VideoWriter('model_all_3D_new.avi', cv.VideoWriter_fourcc('M','J','P','G'), 5, (frame_width, frame_height))

def mean_yellow():    
    yellow_data = 140
    datapoints_yellow = []
    for i in range(0, yellow_data):
        string_path = path_yellow +"/yellow"+str(i)+".jpg"
        img = cv.imread(string_path)
        height, width, channels = img.shape[0], img.shape[1], img.shape[2]
        img = np.reshape(img, (height * width, channels))
        for pixels in range(img.shape[0]):
            datapoints_yellow.append(img[pixels, :])
    return datapoints_yellow

def mean_orange():    
    orange_data = 111
    datapoints_orange = []
    for i in range(0, orange_data):
        string_path = path_orange +"/orange"+str(i)+".jpg"
        img = cv.imread(string_path)
        height, width, channels = img.shape[0], img.shape[1], img.shape[2]
        img = np.reshape(img, (height * width, channels))
        for pixels in range(img.shape[0]):
            datapoints_orange.append(img[pixels, :])
    return datapoints_orange

def mean_green():    
    green_data = 28
    datapoints_green = []
    for i in range(0, green_data):
        string_path = path_green +"/green"+str(i)+".jpg"
        img = cv.imread(string_path)
        height, width, channels = img.shape[0], img.shape[1], img.shape[2]
        img = np.reshape(img, (height * width, channels))
        for pixels in range(img.shape[0]):
            datapoints_green.append(img[pixels, :])
    return datapoints_green

def look_at_histogram():
    hist_size = [255]
    hist_range = [0,256]

    yellow_data = 140
    histogram_r = np.zeros((255,1))
    histogram_g = np.zeros((255,1))
    histogram_b = np.zeros((255,1))

    orange_data = 111
    histogram_ro = np.zeros((255,1))
    histogram_go = np.zeros((255,1))
    histogram_bo = np.zeros((255,1))

    green_data = 28
    histogram_rg = np.zeros((255,1))
    histogram_gg = np.zeros((255,1))
    histogram_bg = np.zeros((255,1))

    for i in range(0, yellow_data):
        string_path = path_yellow + "/yellow"+str(i)+".jpg"
        img = cv.imread(string_path) 
        color = ("b","g","r")
        for j,c in enumerate(color):
            if c == "b":
                temp_b = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_b = np.column_stack((histogram_b, temp_b))

            if c == "g":
                temp_g = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_g = np.column_stack((histogram_g, temp_g))

            if c == "r":
                temp_r = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_r = np.column_stack((histogram_r, temp_r))

    for i in range(0, orange_data):
        string_path = path_orange + "/orange"+str(i)+".jpg"
        img = cv.imread(string_path) 
        color = ("b","g","r")
        for j,c in enumerate(color):
            if c == "b":
                temp_b = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_bo = np.column_stack((histogram_bo, temp_b))

            if c == "g":
                temp_g = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_go = np.column_stack((histogram_go, temp_g))

            if c == "r":
                temp_r = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_ro = np.column_stack((histogram_ro, temp_r))

    for i in range(0, green_data):
        string_path = path_green + "/green"+str(i)+".jpg"
        img = cv.imread(string_path) 
        color = ("b","g","r")
        for j,c in enumerate(color):
            if c == "b":
                temp_b = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_bg = np.column_stack((histogram_bg, temp_b))

            if c == "g":
                temp_g = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_gg = np.column_stack((histogram_gg, temp_g))

            if c == "r":
                temp_r = cv.calcHist([img],[j],None,hist_size,hist_range, accumulate = 1)
                histogram_rg = np.column_stack((histogram_rg, temp_r))

    #sum across the columns, divide by total number of observations
    histogram_avg_b = np.sum(histogram_b, axis=1) / (yellow_data)
    histogram_avg_g = np.sum(histogram_g, axis=1) / (yellow_data)
    histogram_avg_r = np.sum(histogram_r, axis=1) / (yellow_data) 

    #sum across the columns, divide by total number of observations
    histogram_avg_bo = np.sum(histogram_bo, axis=1) / (orange_data)
    histogram_avg_go = np.sum(histogram_go, axis=1) / (orange_data)
    histogram_avg_ro = np.sum(histogram_ro, axis=1) / (orange_data) 

    #sum across the columns, divide by total number of observations
    histogram_avg_bg = np.sum(histogram_bg, axis=1) / (green_data)
    histogram_avg_gg = np.sum(histogram_gg, axis=1) / (green_data)
    histogram_avg_rg = np.sum(histogram_rg, axis=1) / (green_data)

    # Uncomment to plot histograms
    plt.title('Average Histogram for Yellow Buoy')
    plt.subplot(3,1,1)
    plt.plot(histogram_avg_b, color = "b")
    plt.subplot(3,1,2)
    plt.plot(histogram_avg_g, color = "g")
    plt.subplot(3,1,3) 
    plt.plot(histogram_avg_r, color = "r")
    plt.figure()

    plt.title('Average Histogram for Orange Buoy')
    plt.subplot(3,1,1)
    plt.plot(histogram_avg_bo, color = "b")
    plt.subplot(3,1,2)
    plt.plot(histogram_avg_go, color = "g")
    plt.subplot(3,1,3)
    plt.plot(histogram_avg_ro, color = "r")
    plt.figure()

    plt.title('Average Histogram for Green Buoy')
    plt.subplot(3,1,1)
    plt.plot(histogram_avg_bg, color = "b")
    plt.subplot(3,1,2)
    plt.plot(histogram_avg_gg, color = "g")
    plt.subplot(3,1,3)
    plt.plot(histogram_avg_rg, color = "r")
    plt.show()


def learn_with_em(xtrain, K, iters):
    n_points, dimen = xtrain.shape
    mean = np.float64(xtrain[np.random.choice(n_points, K, False), :])
    # # print(mean)  
    covar = [np.random.randint(1,255) * np.eye(dimen)] * K# [150*np.eye(d)] * K
    # print(covar)
    for i in range(K):
        covar[i]=np.multiply(covar[i],np.random.rand(dimen,dimen))
    

    max_bound = 0.0001
    pi_k = [1./K] * K
    prob_cluster__given_x = np.zeros((n_points, K))

    log_likelihoods_array = []
    
    while len(log_likelihoods_array) < iters:
        # Expectation Step
        # print(len(log_likelihoods_array))
        for k in range(K):
            tmp = pi_k[k] * mvn.pdf(xtrain, mean[k], covar[k], allow_singular=True)
            prob_cluster__given_x[:,k]=tmp.reshape((n_points,))

        log_likelihood = np.sum(np.log(np.sum(prob_cluster__given_x, axis = 1)))

        print ("{0} -> {1}".format(len(log_likelihoods_array),log_likelihood))

        log_likelihoods_array.append(log_likelihood)
        prob_cluster__given_x = (prob_cluster__given_x.T / np.sum(prob_cluster__given_x, axis = 1)).T
         
        N_ks = np.sum(prob_cluster__given_x, axis = 0)
        
        # Maximization Step
        for k in range(K):
            # temp = math.fsum(prob_cluster__given_x[:,k])
            mean[k] = 1. / N_ks[k] * np.sum(prob_cluster__given_x[:, k] * xtrain.T, axis = 1).T
            diff_x_mean = xtrain - mean[k]
            covar[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(diff_x_mean.T,  prob_cluster__given_x[:, k]), diff_x_mean))
            pi_k[k] = 1. / n_points * N_ks[k]
        if len(log_likelihoods_array) < 2 : continue
        if np.abs(log_likelihood - log_likelihoods_array[-2]) < max_bound or len(log_likelihoods_array) > 1000: break

    return mean, covar, pi_k, log_likelihoods_array

def all_buoys_visual(trained_mean_y, trained_covar_y, train_pi_k_y, trained_mean_o, trained_covar_o, train_pi_k_o, trained_mean_g, trained_covar_g, train_pi_k_g, Ky, Ko, Kg):
    print("Frame Reading started..")
    while (vid.isOpened()):
        ret,frame = vid.read()
        if frame is not None:
            frame_orig = frame
            if ret == True:            
                height, width, channels = frame.shape[0], frame.shape[1], frame.shape[2]
                frame = np.reshape(frame, (height * width, channels))
                
                # Yellow Buoy
                prob_of_yellow_buoy = np.zeros((height * width, Ky))
                yellow_likelihood = np.zeros((height * width, Ky))

                for k in range(0, Ky):
                    prob_of_yellow_buoy[:, k] = train_pi_k_y[k] * mvn.pdf(frame, trained_mean_y[k], trained_covar_y[k])
                    yellow_likelihood = prob_of_yellow_buoy.sum(1)

                yellow_prob = np.reshape(yellow_likelihood, (height, width))
                yellow_prob[yellow_prob > np.max(yellow_prob)/5.0] = 255
                
                mask_image_yellow = np.zeros((height, width, channels), np.uint8)
                mask_image_yellow[:,:,0] = yellow_prob
                mask_image_yellow[:,:,1] = yellow_prob
                mask_image_yellow[:,:,2] = yellow_prob
                kernel_ellipse_yellow = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
                np.array([[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]], dtype=np.uint8)
                
                dilated_yellow = cv.dilate(mask_image_yellow, kernel_ellipse_yellow, iterations = 1)
                edges_yellow = cv.Canny(dilated_yellow, 50, 255)

                contours_image_yellow, _ = cv.findContours(edges_yellow, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                (contour_sorted_yellow, bounds_yellow) = contours.sort_contours(contours_image_yellow)

                hull_yellow = cv.convexHull(contour_sorted_yellow[0])
                (x_yellow, y_yellow), radius_yellow = cv.minEnclosingCircle(hull_yellow)
                
                if radius_yellow > 3:
                    cv.circle(frame_orig, (int(x_yellow), int(y_yellow) + 4), int(radius_yellow + 2), (0,255,255), 4)
                # cv.imshow("Final Yellow", frame_orig)
                # print("Yellow Buoy processed..")

                # Orange Buoy
                prob_of_orange_buoy = np.zeros((height * width, Ko))
                orange_likelihood = np.zeros((height * width, Ko))

                for k in range(0, Ko):
                    prob_of_orange_buoy[:, k] = train_pi_k_o[k] * mvn.pdf(frame, trained_mean_o[k], trained_covar_o[k])
                    orange_likelihood = prob_of_orange_buoy.sum(1)

                orange_prob = np.reshape(orange_likelihood, (height, width))
                orange_prob[orange_prob > np.max(orange_prob)/2.0] = 255
                
                
                mask_image_orange =np.zeros((height, width, channels), np.uint8)
                mask_image_orange[:,:,0] = orange_prob
                mask_image_orange[:,:,1] = orange_prob
                mask_image_orange[:,:,2] = orange_prob
                kernel_ellipse_orange = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
                np.array([[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]], dtype=np.uint8)
                
                dilated_orange = cv.dilate(mask_image_orange, kernel_ellipse_orange, iterations = 1)
                edges_orange = cv.Canny(dilated_orange, 50, 255)

                contours_image_orange, _ = cv.findContours(edges_orange, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                (contour_sorted_orange, bounds_orange) = contours.sort_contours(contours_image_orange)

                hull_orange = cv.convexHull(contour_sorted_orange[0])
                (x_orange, y_orange), radius_orange = cv.minEnclosingCircle(hull_orange)
                
                if radius_orange > 9 and ((x_orange > 120 and y_orange > 320) or (x_orange > 350 and y_orange > 200)):
                    cv.circle(frame_orig, (int(x_orange), int(y_orange)), int(radius_orange + 1), (36,171,255), 4)
                # cv.imshow("Final Orange", frame_orig)
                # print("Orange Buoy processed..")

                # Green Buoy
                prob_of_green_buoy = np.zeros((height * width, Kg))
                green_likelihood = np.zeros((height * width, Kg))

                for k in range(0, Kg):
                    prob_of_green_buoy[:, k] = train_pi_k_g[k] * mvn.pdf(frame, trained_mean_g[k], trained_covar_g[k])
                    green_likelihood = prob_of_green_buoy.sum(1)

                green_prob = np.reshape(green_likelihood, (height, width))
                green_prob[green_prob > np.max(green_prob)/2.0] = 255
                mask_image_green =np.zeros((height, width, channels), np.uint8)
                mask_image_green[:,:,0] = green_prob
                mask_image_green[:,:,1] = green_prob
                mask_image_green[:,:,2] = green_prob
                kernel_ellipse_green = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
                np.array([[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]], dtype=np.uint8)
                
                dilated_green = cv.dilate(mask_image_green, kernel_ellipse_green, iterations = 1)
                edges_green = cv.Canny(dilated_green, 50, 255)

                contours_image_green, _ = cv.findContours(edges_green, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                (contour_sorted_green, bounds_green) = contours.sort_contours(contours_image_green)

                hull_green = cv.convexHull(contour_sorted_green[0])
                (x_green, y_green), radius_green = cv.minEnclosingCircle(hull_green)
                
                if radius_green > 9 and ((x_green > 320 and y_green > 300) or (x_green > 350 and y_green > 200)):
                    cv.circle(frame_orig, (int(x_green), int(y_green)), int(radius_green + 1), (50,240,91), 4)
                # print("Green Buoy processed..")

                cv.imshow("Final Frame", frame_orig)

                if flag == "1":
                    out.write(frame_orig)
                k = cv.waitKey(15) & 0xff
                if k == 27:
                    break

        else:
            break
    vid.release()
    if flag == "1":
        print("Output video saved!")
        out.release()
    else:
        print("Completed.")

# Cluster required for Yellow
Ky = 7
# Cluster required for Orange
Ko = 4
# Cluster required for Green
Kg = 5
# Max iterations
iterations = 1500

if val == "1":
    # Uncomment to see the Average Histogram
    look_at_histogram()
    print("Optimal K would be yellow: 7, orange: 4, green: 5.")

if val == "2":
    # # Uncomment this
    mean_yellow_pts = mean_yellow()
    print("For Yellow")
    trained_mean_y, trained_covar_y, train_pi_k_y, log_likelihoods_array_y = learn_with_em(np.array(mean_yellow_pts), Ky, iterations)
    np.save('mean_yellow_all.npy', trained_mean_y)
    np.save('covar_yellow_all.npy', trained_covar_y)
    np.save('weights_yellow_all.npy', train_pi_k_y)
    
    # # Uncomment this
    mean_orange_pts = mean_orange()
    print("For Orange")
    trained_mean_o, trained_covar_o, train_pi_k_o, log_likelihoods_array_o = learn_with_em(np.array(mean_orange_pts), Ko, iterations)
    np.save('mean_orange_all.npy', trained_mean_o)
    np.save('covar_orange_all.npy', trained_covar_o)
    np.save('weights_orange_all.npy', train_pi_k_o)    

    # # Uncomment this
    mean_green_pts = mean_green()
    print("For Green")
    trained_mean_g, trained_covar_g, train_pi_k_g, log_likelihoods_array_g = learn_with_em(np.array(mean_green_pts), Kg, iterations)
    np.save('mean_green_all.npy', trained_mean_g)
    np.save('covar_green_all.npy', trained_covar_g)
    np.save('weights_green_all.npy', train_pi_k_g)

    plt.plot(log_likelihoods_array_y)
    plt.title('Log Likelihood(Yellow) vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood(Yellow)')
    plt.figure()

    plt.plot(log_likelihoods_array_o)
    plt.title('Log Likelihood(Orange) vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood(Orange)')
    plt.figure()

    plt.plot(log_likelihoods_array_g)
    plt.title('Log Likelihood(Green) vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood(Green)')
    plt.show()

    print("EM trained, parameters saved into binary")

if val == "3":
    trained_mean_y = np.load('mean_yellow_all.npy') 
    trained_covar_y = np.load('covar_yellow_all.npy')
    train_pi_k_y = np.load('weights_yellow_all.npy')
    print(trained_mean_y, trained_covar_y, train_pi_k_y)

    trained_mean_o = np.load('mean_orange_all.npy') 
    trained_covar_o = np.load('covar_orange_all.npy')
    train_pi_k_o = np.load('weights_orange_all.npy')
    print(trained_mean_o, trained_covar_o, train_pi_k_o)

    trained_mean_g = np.load('mean_green_all.npy') 
    trained_covar_g = np.load('covar_green_all.npy')
    train_pi_k_g = np.load('weights_green_all.npy')
    print(trained_mean_g, trained_covar_g, train_pi_k_g)

    print("EM trained with saved parameters..")
    # yellow_buoy_visual(trained_mean_y, trained_covar_y, train_pi_k_y, Ky)
    # orange_buoy_visual(trained_mean_o, trained_covar_o, train_pi_k_o, Ko)
    # green_buoy_visual(trained_mean_g, trained_covar_g, train_pi_k_g, Kg)

    all_buoys_visual(trained_mean_y, trained_covar_y, train_pi_k_y, trained_mean_o, trained_covar_o, train_pi_k_o, trained_mean_g, trained_covar_g, train_pi_k_g, Ky, Ko, Kg)
cv.destroyAllWindows()