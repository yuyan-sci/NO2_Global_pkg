from OSM_pkg.Assemble_Func import derive_raster_RoadDensityMap_forRegions,derive_raster_RoadDensityMap_forGlobal
from config import Continent_list, Region_list
import argparse


parser=argparse.ArgumentParser()
parser.add_argument('--Continent',type=str)
parser.add_argument('--Country',type=str)
parser.add_argument('--regions_list',nargs='+',type=str)
parser.add_argument('--Entry_List_forRegional_RoadDensityMap',nargs='+',type=str)


derive_regional_RoadDensityMap_switch  = False
Continent_froRegional_RoadDensityMap   = parser.parse_args().Continent
Country_froRegional_RoadDensityMap     = parser.parse_args().Country
OSM_YEAR_froRegional_RoadDensityMap    = 2025
Region_List_forRegional_RoadDensityMap = parser.parse_args().regions_list
Entry_List_forRegional_RoadDensityMap  = parser.parse_args().Entry_List_forRegional_RoadDensityMap


derive_Global_RoadDensityMap_switch        = True
Continent_List_forGlobal_RoadDensityMap    = Continent_list
Region_List_forGlobal_RoadDensityMap       = Region_list
Entry_List_forGlobal_RoadDensityMap = ['primary','secondary', 'motorway', 'trunk', 'tertiary', 'residential','unclassified']
OSM_YEAR = 2025   

if __name__ == '__main__':
    if derive_regional_RoadDensityMap_switch:
        derive_raster_RoadDensityMap_forRegions(
            Continent=Continent_froRegional_RoadDensityMap,
            Region_List=Region_List_forRegional_RoadDensityMap,
            OSM_YEAR=OSM_YEAR_froRegional_RoadDensityMap,
            entries_list=Entry_List_forRegional_RoadDensityMap,
            head_index='fclass',
            resolution=0.01
        )

        
    if derive_Global_RoadDensityMap_switch:
        derive_raster_RoadDensityMap_forGlobal(
            Continent_List=Continent_List_forGlobal_RoadDensityMap,
            Region_List=Region_List_forGlobal_RoadDensityMap,
            entries_list=Entry_List_forGlobal_RoadDensityMap,
            YEAR=OSM_YEAR
        )