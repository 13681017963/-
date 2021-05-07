"""
海温场-EOF分析
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
from src.config import algorithm
from src.config import diag_service
from src.config import diag_config


#获取图形标题
def get_title(num):
    title = "海温EOF 第"+str(num+1)+"空间模态"
    return title

#开始绘图
def draw_map(start,end,res,lat,lon,path):
    # 开始绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    m = Basemap(resolution='l', area_thresh=10000, projection='cyl', llcrnrlat=int(slat), urcrnrlat=int(elat),
                llcrnrlon=int(slon), urcrnrlon=int(elon))
    cmap1,color_arr = diag_config.get_eof_cmap_arr()
    cm.register_cmap(cmap=cmap1)
    x, y = m(*np.meshgrid(lon, lat))
    for i in range(len(res)):
        plt.figure(figsize=(10,8))
        m.drawparallels(np.arange(int(slat), int(elat), 20), labels=[1, 0, 0, 0], fontsize=12, linewidth=0.01)
        m.drawmeridians(np.arange(int(slon), int(elon), 30), labels=[0, 0, 0, 1], fontsize=12, linewidth=0.01)
        # 填充颜色
        CS2 = m.contourf(x, y, res[i], color_arr, cmap=cm.get_cmap('own'), extent='both')
        #添加色度条
        m.colorbar(CS2,location='bottom',pad="15%")
        #绘制海岸线
        m.drawcoastlines(linewidth=0.5)
        ##设置标题
        plt.title(get_title(i), size=17)
        plt.savefig(path + "eof_oce_" + str(i + 1) + ".png", dpi=300)
        print("ok")

if __name__ == '__main__':
    # - - - - - - - - - - - 参数区域- - - - - - - - - - -
    start = "199106"  # 开始时间
    end = "201108"  # 结束时间
    slat = -67  # 起始纬度
    elat = 67  # 终止纬度
    slon = 40  # 起始经度
    elon = 260  # 终止经度
    num = 5  # 模态个数
    filepath = "D:\\bcacfs\\nc\\ocean\\sst.mnmean.v3.nc"
    pattern = 0
    qua = 1  # 1：距平，2：标准化，3：原始值
    out_path = "D:\\"


    # - - - - - - - - - - - 计算区域 - - - - - - - - - - - - -
    t_year = algorithm.get_year_result(start, end)
    seqo,lat,lon = diag_service.getOceList(start,end,slat,elat,slon,elon,filepath,pattern)
    seq = algorithm.get_result_list(seqo,qua,t_year)
    eofs,pcs = algorithm.eof(seq,num)

    # - - - - - - - - - - - 绘图区域 - - - - - - - - - - - - - -
    draw_map(start, end, eofs, lat, lon, out_path)
