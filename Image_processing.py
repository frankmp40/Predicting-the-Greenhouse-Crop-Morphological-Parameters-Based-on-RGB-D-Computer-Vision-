import pandas as pd
import numpy as np
from pathlib import Path
from functions import EXI_cal
import cv2

# Define number of images
MAX_IMG_NO = 391

class image_process(object):
    def __init__(self):
        self.dataframe = None

    def execute(self):
        sample_list = range(1,MAX_IMG_NO+1)
        img_id = []
        leaf_pixel_ratio = []
        edge_pixel_ratio = []
        exg_list = []
        exr_list = []
        vari_list = []

        for i in sample_list:
            img_no = i
            print('Img no {} processed'.format(i))
            my_file = Path('Processed_images\extracted_img\extract_{}.png'.format(img_no))
            if my_file.is_file():
                img_rgb = cv2.imread('Processed_images\extracted_img\extract_{}.png'.format(img_no))
                img_canny = cv2.imread('Processed_images\canny_edge\canny_{}.png'.format(img_no))

                # Get ratio of leaf pixel and edge
                non_black_canny_ratio, non_black_rgb_ratio,\
                exg, exr, vari = self.processing_methods(img_canny, img_rgb)

                # Appending to list
                img_id.append(img_no)
                leaf_pixel_ratio.append(non_black_rgb_ratio)
                edge_pixel_ratio.append(non_black_canny_ratio)
                exg_list.append(exg)
                exr_list.append(exr)
                vari_list.append(vari)

        self.dataframe = pd.DataFrame({'Img_id':img_id, 'Leaf_pixel_ratio':leaf_pixel_ratio, 'Edge_pixel_ratio':edge_pixel_ratio,
                                       'EXG':exg_list, 'EXR':exr_list, 'VARI':vari_list})

        # self.dataframe = pd.DataFrame({'Img_id':img_id, 'Leaf_pixel_ratio':leaf_pixel_ratio, 'Edge_pixel_ratio':edge_pixel_ratio,
        #                                })

        return self.dataframe

    # Calculate the ratio of leaf pixcel and edge pixcel in the image
    def processing_methods(self, img_canny, img_rgb):
        # Count non black pixels in the extracted rgb and canny images
        sought = [0, 0, 0]
        black_rgb_img = np.count_nonzero(np.all(img_rgb == sought, axis=2))
        black_canny_img = np.count_nonzero(np.all(img_canny == sought, axis=2))
        non_black_rgb_ratio = 1 - black_rgb_img / (img_rgb.shape[0] * img_rgb.shape[1])
        non_black_canny_ratio = 1 - black_canny_img / (img_canny.shape[0] * img_canny.shape[1])
        # Calculate EXG, EXB, EXR

        exg, exr, vari = self.reflectance_average(img_rgb)
        return non_black_canny_ratio, non_black_rgb_ratio, exg, exr, vari

    def reflectance_average(self, img_rgb):
        EXG = []
        EXR = []
        VARI = []
        for x in range(img_rgb.shape[0]):
            for y in range(img_rgb.shape[1]):
                B = int(img_rgb[x,y,0]); G = int(img_rgb[x,y,1]); R = int(img_rgb[x,y,2])
                # Only canopy area will be calculated
                if (B+G+R)!=0:
                    exg,exr,vari = EXI_cal(B, G, R)
                    EXG.append(exg)
                    EXR.append(exr)
                    VARI.append(vari)
        # Calculate mean of the EXG of the canopy area
        return np.array(EXG).mean(), np.array(EXR).mean(), np.array(VARI).mean()

