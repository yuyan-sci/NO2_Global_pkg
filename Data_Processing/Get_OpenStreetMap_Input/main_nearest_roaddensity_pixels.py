import numpy as np
from OSM_pkg.Assemble_Func import derive_NearestPixels_RoadDensity
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-R','--RoadEntry',type=str)
roadentry = parser.parse_args().RoadEntry
if __name__ == '__main__':
    YEAR=2025
    derive_NearestPixels_RoadDensity(YEAR=YEAR,entry_name=roadentry)
    
