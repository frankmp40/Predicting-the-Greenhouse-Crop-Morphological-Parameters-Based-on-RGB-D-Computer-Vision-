import pandas as pd
from Ground_Truth import load_data
from Image_processing import image_process
import matplotlib.pyplot as plt
from functions import plot_relationship
import seaborn as sn

# Read ground true data
data_GT = load_data().execute()

# Process extracted images get input data
input_img_task = image_process()
data_input = input_img_task.execute()

# Join the ground truth data and the input data
data_full = pd.merge(data_input, data_GT, on='Img_id')
data_full['DMC'] = data_full['DryWeightShoot']/data_full['FreshWeightShoot']
data_full.pop('RGBImage')
data_full.pop('DepthInformation')

# Save dataset
data_full.to_csv('Processed_data/data_analysis.csv', index=None)

# # Plot relationships
# # All lettuce type
# plot_relationship(data_full['Leaf_pixel_ratio'], data_full, 'All Type - LA_pixel_ratio', 'LA_pixel_ratio')
# plot_relationship(data_full['Edge_pixel_ratio'], data_full, 'All Type - Edge_pixel_ratio', 'Edge_pixel_ratio')
# plot_relationship(data_full['EXG'], data_full, 'All Type - EXG', 'EXG')
# plot_relationship(data_full['EXR'], data_full, 'All Type - EXR', 'EXR')
# plot_relationship(data_full['VARI'], data_full, 'All Type - VARI', 'VARI')

# # Aphylion type
# data_aphylion = data_full[data_full['Variety']=='Aphylion']
# plot_relationship(data_aphylion['Leaf_pixel_ratio'], data_aphylion, 'Aphylion - LA_pixel_ratio', 'LA_pixel_ratio')
# plot_relationship(data_aphylion['Edge_pixel_ratio'], data_aphylion, 'Aphylion - Edge_pixel_ratio', 'Edge_pixel_ratio')
# plot_relationship(data_aphylion['EXG'], data_aphylion, 'Aphylion - EXG', 'EXG')
# plot_relationship(data_aphylion['EXR'], data_aphylion, 'Aphylion - EXR', 'EXR')
# plot_relationship(data_aphylion['VARI'], data_aphylion, 'Aphylion - VARI', 'VARI')

# # # Salanova type
# data_salanova= data_full[data_full['Variety']=='Salanova']
# plot_relationship(data_salanova['Leaf_pixel_ratio'], data_salanova, 'Salanova - LA_pixel_ratio', 'LA_pixel_ratio')
# plot_relationship(data_salanova['Edge_pixel_ratio'], data_salanova, 'Salanova - Edge_pixel_ratio', 'Edge_pixel_ratio')
# plot_relationship(data_salanova['EXG'], data_salanova, 'Salanova - EXG', 'EXG')
# plot_relationship(data_salanova['EXR'], data_salanova, 'Salanova - EXR', 'EXR')
# plot_relationship(data_salanova['VARI'], data_salanova, 'Salanova - VARI', 'VARI')

# # Satine type
# data_satine = data_full[data_full['Variety']=='Satine']
# plot_relationship(data_satine['Leaf_pixel_ratio'], data_satine, 'Satine - LA_pixel_ratio', 'LA_pixel_ratio')
# plot_relationship(data_satine['Edge_pixel_ratio'], data_satine, 'Satine - Edge_pixel_ratio', 'Edge_pixel_ratio')
# plot_relationship(data_satine['EXG'], data_satine, 'Satine - EXG', 'EXG')
# plot_relationship(data_satine['EXR'], data_satine, 'Satine - EXR', 'EXR')
# plot_relationship(data_satine['VARI'], data_satine, 'Satine - VARI', 'VARI')

# # Lugano type
# data_lugano = data_full[data_full['Variety']=='Lugano']
# plot_relationship(data_lugano['Leaf_pixel_ratio'], data_lugano, 'Lugano - LA_pixel_ratio', 'LA_pixel_ratio')
# plot_relationship(data_lugano['Edge_pixel_ratio'], data_lugano, 'Lugano - Edge_pixel_ratio', 'Edge_pixel_ratio')
# plot_relationship(data_lugano['EXG'], data_lugano, 'Lugano - EXG', 'EXG')
# plot_relationship(data_lugano['EXR'], data_lugano, 'Lugano - EXR', 'EXR')
# plot_relationship(data_lugano['VARI'], data_lugano, 'Lugano - VARI', 'VARI')

# # Plot features heatmap
# data_heatmap = data_input.loc[:, data_input.columns!='Img_id']
# corrMatrix = data_heatmap.corr()
# sn.heatmap(corrMatrix, annot=True)
