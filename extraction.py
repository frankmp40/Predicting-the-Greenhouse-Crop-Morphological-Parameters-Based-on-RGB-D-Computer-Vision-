import open3d as o3d
import numpy as np
import cv2
import random
from pathlib import Path

# Define number of images
MAX_IMG_NO = 391

def depth_array_limit_filter(data, limit):
    data_new = np.zeros(data.shape)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if data[row][col] > limit:
                data_new[row][col] = limit
            else:
                data_new[row][col] = data[row][col]
    
    return data_new

def depth_error_correct(depth):
    data_new = np.zeros(depth.shape)
    for row in range(depth.shape[0]):
        for col in range(depth.shape[1]):
            if depth[row][col] < 700:
                data_new[row][col] = 1500
            else:
                data_new[row][col] = depth[row][col]
    return data_new

def depth_to_rgb_gray(depth):
    upper = depth.max()
    lower = 700
    rgb = np.zeros(depth.shape)
    
    for row in range(depth.shape[0]):
        for col in range(depth.shape[1]):
            value = depth[row][col]
            rgb[row][col] = 255*(value-lower)/(upper-lower)
            
    return rgb

def random_sample_list(size):
    randomlist = []
    for i in range(0,size):
        n = random.randint(1,391)
        randomlist.append(n)
    print(randomlist)

    return randomlist


# generate random list
# sample_list = random_sample_list(30)
sample_list = range(1,MAX_IMG_NO+1)

for i in sample_list:
    img_no = i
    my_file = Path('DepthImages/Depth_{}.png'.format(img_no))
    if my_file.is_file():

        img = cv2.imread('RGBImages/RGB_{}.png'.format(img_no))
        depth_image = o3d.io.read_image('DepthImages/Depth_{}.png'.format(img_no))

        # read array
        depth_array = np.asarray(depth_image)

        # remove outliers in depth array
        depth_array_process = depth_array_limit_filter(depth_array, 1500)

        # depth correction
        depth_array_process = depth_error_correct(depth_array_process)

        # convert to rgb
        depth_img = depth_to_rgb_gray(depth_array_process)

        # apply rgb_depth filter
        thresh = 60
        ret, thresh_d = cv2.threshold(depth_img,thresh,255,cv2.THRESH_BINARY_INV)
        thresh_d = thresh_d.astype('uint8')

        # hsv & thresholding
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        thresh = 50
        ret, thresh_s = cv2.threshold(s,thresh,255,cv2.THRESH_BINARY)

        # BGR thresholding - remove orange colour
        b,g,r = cv2.split(img)
        thresh = 180
        ret, thresd_r = cv2.threshold(r, thresh, 255, cv2.THRESH_BINARY_INV)

        # bitwise
        combine = thresh_s & thresh_d & thresd_r
        combine_crop = combine[200:1000, 680:1460]

        # contour mask
        contours, hierarchy = cv2.findContours(combine_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        area = []
        for k in range(len(contours)):
        	area.append(cv2.contourArea(contours[k]))
        max_idx = np.argmax(np.array(area))

        mask = np.zeros(img[200:1000, 680:1460].shape, np.uint8)
        cv2.drawContours(mask, contours, max_idx, (255,255,255), cv2.FILLED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # extract image by mask
        extacted_img = cv2.bitwise_and(img[200:1000, 680:1460], img[200:1000, 680:1460], mask=mask)

        # add canny edge
        edges = cv2.Canny(extacted_img, 100, 70)

        # save imgs
        cv2.imwrite('Processed_images/extracted_img/extract_{}.png'.format(img_no), extacted_img)
        cv2.imwrite('Processed_images/canny_edge/canny_{}.png'.format(img_no), edges)
        cv2.imwrite('Processed_images/depth/'
                    '{}.png'.format(img_no), depth_img)
        print('img_no: {} extracted'.format(img_no))

    else:
        print('image {} not exist'.format(i))






