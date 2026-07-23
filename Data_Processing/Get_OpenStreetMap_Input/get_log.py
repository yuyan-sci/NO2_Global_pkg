from OSM_pkg.Assemble_Func import *


if __name__ == '__main__':
    YEAR=2025
    Area='Global'
    get_log(YEAR=YEAR)
    for buffer in [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11]:
        add_buffer(buffer=buffer)