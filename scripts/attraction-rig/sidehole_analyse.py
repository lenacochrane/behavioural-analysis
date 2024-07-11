import pandas as pd
import numpy as np
import os 


class Side_hole_analysis:

    def __init__(self, directory):

        self.directory = directory 
        self.coordinates_files = []
        self.track_files = []
        self.coordinates()
        self.tracks()

    def coordinates(self, directory):
        # 2024-05-20_16-08-22_td1_hole.csv
        self.coordinate_files = [f for f in os.listdir(self.directory) if f.endswith('hole.csv')]

    def tracks(self, directory):
        # 2024-04-30_14-31-44_td5.000_2024-04-30_14-31-44_td5.analysis.csv
        self.track_files = [f for f in os.listdir(self.directory) if f.endswith('analysis.csv')]

    def side_hole_for_video(self, coordinates_files, track_files):

        






        





# i will have track csv and holes csv coordinates

# defintion which fills in the coordinates of the hole and removes it from the csv file idk like any larvae inside is not counted - 


# # things i would want to use in every analysis which is useful
# #   - iterate over every file in a directory 
#     - - THEN SCRIPT FOR ANALYSIS NOT IN HOLE OR IN HOLE - count number of tracks per frame but not those in the hole + CERTAIN RADIUS
# - RETURNS TO HOLE? - tracks which appear in certain radius and re enter the radis - like a counter 
# - DISTANCE FROM HOLE - for every larvae count the distance from the hole - take the avergae coordinates of the hoel 



