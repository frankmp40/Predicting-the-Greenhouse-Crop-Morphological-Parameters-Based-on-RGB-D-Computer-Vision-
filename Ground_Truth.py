import json
import pandas as pd

# Define number of images
MAX_IMG_NO = 391

class load_data(object):

    def __init__(self):
        self.data_json = None
        self.dataframe = None

    def read_json(self):
        # load json file
        with open('GroundTruth/GroundTruth_All_388_Images.json','r') as f:
            self.data_json = json.load(f)


    def create_dataframe(self):
        # Create Json Dataframe
        img_id = []
        variety = []
        rgbImage = []
        debthInformation = []
        freshWeightShoot = []
        dryWeightShoot = []
        height = []
        diameter = []
        leafArea = []

        for i in range(1, MAX_IMG_NO+1):
            # Image332 has a problem
            if i!=332:
                # The max image number, some image is missing, ignore missing image file
                file_name = 'Image' + '%s'%i
                try:
                    data_list = list(self.data_json['Measurements'][file_name].values())
                    img_id.append(i)
                    variety.append(data_list[0])
                    rgbImage.append(data_list[1])
                    debthInformation.append(data_list[2])
                    freshWeightShoot.append(data_list[3])
                    dryWeightShoot.append(data_list[4])
                    height.append(data_list[5])
                    diameter.append(data_list[6])
                    leafArea.append(data_list[7])
                except:
                    pass

        # Generate the dataframe
        self.dataframe = pd.DataFrame({'Img_id':img_id, 'Variety':variety, 'RGBImage':rgbImage, 'DepthInformation':debthInformation,
                                    'FreshWeightShoot':freshWeightShoot, 'DryWeightShoot':dryWeightShoot,
                                    'Height':height, 'Diameter':diameter, 'LeafArea':leafArea})
        return self.dataframe

    def execute(self):
        self.read_json()
        dataframe = self.create_dataframe()
        return dataframe