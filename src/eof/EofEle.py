"""
要素-EOF分析
"""

import numpy as np
from src.config import station_config
from src.config import algorithm
from src.config import diag_config
from src.config import diag_service
import dclimate.d_std_lib as  d_std_lib
import os

#获取图形标题
def get_title(area,var,num):
    title = station_config.get_area_name(area) + diag_config.get_diag_name(
        var) + "EOF第"+str(num+1)+"空间模态"
    print(title)
    return title


#读取res站点数据开始绘图
def draw_map(var, start, end, sta_order_list, res, area, num):
    for i in range(num-1):
        result = res[i]
        Region_ID = area
        res_list = algorithm.make_station_list(sta_order_list,result)
        # OutPicFile1 = "d:\\eof_" + var + "_" + area +"_"+str(i+1)+ ".png"  # 图片输出路径
        OutPicFile1 = "./" + "eof_" + var + "_" + area + "_" + str(i + 1) + ".png"
        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        LevelFile = root_path + '\\config\\LEV\\eof\\maplev_eof_ele.LEV'
        Region_Dict2 = algorithm.get_RegionID_by_XML(root_path +
                                                     '\\config\\sky_region_config_utf8_2.xml', Region_ID)
        title = get_title(area, var, i)
        d_std_lib.DrawMapMain_XML_CFG(Region_Dict2, res_list, Levfile=LevelFile, \
                                      Title=title, imgfile=OutPicFile1,
                                      bShowImage=False, bDrawNumber=False, bDrawColorbar=True,
                                      format1='%1d')  # ,Title='')
        print('ok')

if __name__ == '__main__':
    # - - - - - - - - - - - 参数区域- - - - - - - - - - -
    var = "tem" #要素
    start = "199106"  # 开始时间
    end = "201108"  # 结束时间
    area = 'bj'  # 绘制区域 北京：bj，天津：tj，京津冀：jjj，内蒙古：nmg，华北：huabei，山西：shanxi
    num = 5  # 模态个数
    # - - - - - - - - - - - 计算区域 - - - - - - - - - - - - -
    t_year = algorithm.get_year_result(start, end)  # 获取年份差
    seqe,sta_order_list = diag_service.getEleList(t_year,area)
    eofs,pcs = algorithm.eof(seqe,num)

    # - - - - - - - - - - - 绘图区域 - - - - - - - - - - - - - -
    draw_map(var, start, end, sta_order_list, eofs, area, num)


