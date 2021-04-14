"""
大气环流-EOF分析
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
from src.config import diag_service
from src.config import diag_config
from src.config import algorithm
import netCDF4 as nc
from eofs.standard import Eof

#获取图形标题
def get_title(lev,var,num):
    title = lev + "hPa" + diag_config.get_diag_name(var) + "EOF 第"+str(num+1)+"空间模态"
    return title

#开始绘图
def draw_map(var,lev,start,end,reglist,lat,lon,path):
    # 开始绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    m = Basemap(resolution='l', area_thresh=10000, projection='cyl', llcrnrlat=int(slat), urcrnrlat=int(elat),
                llcrnrlon=int(slon), urcrnrlon=int(elon))
    cmap1,color_arr = diag_config.get_eof_cmap_arr()
    cm.register_cmap(cmap=cmap1)
    for i in range(len(reglist)):
        plt.figure(figsize=(10,8))
        x, y = m(*np.meshgrid(lon, lat))
        m.drawparallels(np.arange(int(slat), int(elat), 10), labels=[1, 0, 0, 0], fontsize=12, linewidth=0.01)
        m.drawmeridians(np.arange(int(slon), int(elon), 30), labels=[0, 0, 0, 1], fontsize=12, linewidth=0.01)
        # 填充颜色
        CS2 = m.contourf(x, y, reglist[i], color_arr, cmap=cm.get_cmap('own'),extend='both')
        # 添加色度条
        m.colorbar(CS2, location='bottom', pad="15%")
        # 绘制海岸线
        m.drawcoastlines(linewidth=0.2)
        ##设置标题
        plt.title(get_title(lev,var,i), size=17)
        plt.savefig(path + "eof_atoms_"+str(i+1)+".png", dpi=300)
        plt.close()
        print("ok")







if __name__ == '__main__':
    # - - - - - - - - - - - 参数区域- - - - - - - - - - -
    var = "hgt" #要素
    lev = "500"
    start = "20150601"  # 开始时间
    end = "20180801"  # 结束时间
    slat = -67  # 起始纬度
    elat = 67  # 终止纬度
    slon = 40  # 起始经度
    elon = 260  # 终止经度
    num = 3 #模态个数
    path = "D:\\bcacfs\\nc\\atmos"  # 环流NC文件路径
    pattern = 0
    qua = 1 #1：距平，2：标准化，3：原始值
    out_path = "D:\\"
    # - - - - - - - - - - - 计算区域 - - - - - - - - - - - - -
    t_year = algorithm.get_year_result(start, end)  # 获取年份差
    seqa,lat,lon = diag_service.getAtmosList(var,lev,start ,end ,t_year,slat,elat,slon,elon,path,pattern)#获取时间段内大气环流原始值序列,以及需要绘图所需经纬度范围
    seq = algorithm.get_result_list(seqa,qua,t_year)
    eofs,pcs = algorithm.eof(seq,num)


    # - - - - - - - - - - - 绘图区域 - - - - - - - - - - - - - -
    draw_map(var,lev,start,end,eofs,lat,lon,out_path)


