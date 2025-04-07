# Predicting-the-Greenhouse-Crop-Morphological-Parameters-Based-on-RGB-D-Computer-Vision

Accurate measurement of crop morphological traits is essential for informed greenhouse management and precision agriculture. Remote sensing technologies, particularly RGB-D-based computer vision, offer promising solutions for automating this process. This project evaluates the effectiveness of RGB-D imaging in predicting key plant parameters, including leaf area, plant height, fresh weight, and canopy diameter, to support data-driven decision-making in greenhouse environments.

## Dataset
Dataset of this project is from the 3rd Autonomous Greenhouse Challenge Online Challenge Lettuce Images:

Hemming, S.S.d.Z., H.F. (Feije); Elings, A. (Anne); bijlaard, monique; Marrewijk, van, Bart; Petropoulou, Anna, r.A.G.C.O.C.L. Images, Editor. 2021: 4TU.ResearchData.

DOI: 10.4121/15023088

## How to use?
extraction.py:

Take the depth images and rgb images as the input, extract lettuce crop image. Output of this code are extracted images and stored to 'Processed_images/'.

data_processing.py: 

Take processed images in 'Processed_images/' as input, extract CV features and make plots.  Output a csv file 'Processed_data/data_analysis.csv' containing CV feature results.

prediction.py:

Take 'Processed_data/data_analysis.csv' as input, user choose data processing method and prediction models, outputs prediction results to 'Processed_data/data_analysis.csv'.



