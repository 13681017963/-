import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from mpl_toolkits.basemap import Basemap
import random
from src.config import station_config
import matplotlib.colors as col
import matplotlib.cm as cm
import os

# 定义过渡颜色映射
startcolor = '#0000ff'  # 红色
midcolor = '#ffffff'  # 白色
endcolor = '#ff0000'  # 蓝色
cmap1 = col.LinearSegmentedColormap.from_list('own', [startcolor, midcolor, endcolor])
cm.register_cmap(cmap=cmap1)


def drawmap(list):
    sta_order_list = []  # 站号顺序序列
    for key in list[0]:
        sta_order_list.append(key)
    ele_list = []  # 需要计算的要素序列
    for item in range(len(list)):
        year_item = np.zeros(len(list))
        num = 0
        for key in list[item]:
            year_item.put(num, list[item][key])
            num = num + 1
        ele_list.append(year_item)
    ave_list = sum(ele_list) / len(ele_list)  # 计算年数总和的平均气温

    # 制作res数组用于绘制地图
    res_list = []
    for id in range(len(sta_order_list)):
        res = np.zeros(3, dtype=np.float)
        station = sta_order_list[id]
        latlon = station_config.get_station_latlon(station)
        latlon_arr = latlon.split(',')
        ele_data = ave_list[id]
        res[0] = latlon_arr[0]
        res[1] = latlon_arr[1]
        res[2] = ele_data
        res_list.append(res)
    res_arr = np.array(res_list)


    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 插值
    lon = res_arr[:,0]
    lat = res_arr[:,1]
    rain_data = res_arr[:,2]
    olon = np.linspace(115.4, 117.5, 88)
    olat = np.linspace(39.4, 41.0, 88)
    olon, olat = np.meshgrid(olon, olat)
    # 插值处理
    func = Rbf(lon, lat, rain_data, function='linear')
    rain_data_new = func(olon, olat)

    # 画图
    plt.rc('font', size=15, weight='bold')
    m = Basemap(projection='cyl', llcrnrlat=39.3, llcrnrlon=115.3, urcrnrlat=41.2, urcrnrlon=117.6)
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    m.readshapefile(root_path+'\\config\\Shape\\beijing\\Beijing_CGCS2000', 'Beijing_CGCS2000.shp', linewidth=1, color='k')

    x, y = m(olon, olat)
    xx, yy = m(lon, lat)
    levels = np.arange(30, 50, 2)

    cf = m.contourf(x, y, rain_data_new, levels=levels, cmap=cmap1)
    m.colorbar(cf, location='bottom', pad="15%", label='摄氏度')
    st = m.scatter(xx - 0.1, yy, c='k', s=10, marker='o')
    for i in range(0, len(xx)):
        plt.text(xx[i], yy[i], 'ss', va='center', fontsize=10)
    # lon_num = np.arange(115.4, 117.5, 0.3)
    # lon_label = ['115.4°', '115.7°', '116.0°', '116.3°', '116.6°', '116.9°', '117.2°', '117.5°E']
    # lat_num = np.arange(26, 39, 2)
    # lat_label = ['39.4°', '39.6°', '39.8°', '40.0°', '40.2°', '40.4°','40.6°','40.8°', '41.0°N']
    # plt.yticks(lat_num, lat_label)
    # plt.xticks(lon_num, lon_label)
    plt.title('测试图')
    plt.show()
    print("ok")

    #plt.savefig('test.png', bbox_inches='tight', dpi=300)



if __name__ == '__main__':
    list = []
    arr1 = {"54399":"-37.16","54433":"-28.31","54511":"-30.07","54514":"-48.63","54398":"-16.32","54412":"-49.56","54416":"-34.69","54419":"-13.53","54421":"-18.56","54424":"-23.54","54431":"-17.15","54594":"-47.31","54501":"-17.91","54505":"-55.41","54596":"-52.52","54597":"-29.85","54406":"-31.88","54499":"-41.44","54410":"-35.51","54513":"-32.24"}
    year = 30
    for y in range(year):
        dict={}
        for key in arr1:
            dict[key] =round(random.uniform(20, 38),1)
        list.append(dict)
    drawmap(list)


