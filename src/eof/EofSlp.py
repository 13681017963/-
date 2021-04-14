"""
大气环流-EOF分析
"""
import matplotlib.pyplot as plt
import numpy as np
import datetime
from mpl_toolkits.basemap import Basemap
import netCDF4 as nc
import matplotlib.cm as cm
from src.config import algorithm
from src.config import diag_service
from src.config import diag_config


#获取图形标题
def get_title(var,num):
    title = "NCEP/NCAR "+diag_config.get_diag_name(var) + "EOF 第"+str(num+1)+"空间模态"
    return title

#开始绘图
def draw_map(var,start,end,reglist,lat,lon,path):
    # 开始绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    m = Basemap(resolution='l', area_thresh=10000, projection='cyl', llcrnrlat=int(slat), urcrnrlat=int(elat),
                llcrnrlon=int(slon), urcrnrlon=int(elon))
    cmap1,color_arr = diag_config.get_eof_cmap_arr()
    cm.register_cmap(cmap=cmap1)
    for i in range(len(reglist)):
        fig = plt.figure(figsize=(12, 6))
        x, y = m(*np.meshgrid(lon, lat))
        m.drawparallels(np.arange(int(slat), int(elat), 10), labels=[1, 0, 0, 0], fontsize=12, linewidth=0.01, fmt=diag_config.formatParallelsDeg)
        m.drawmeridians(np.arange(int(slon), int(elon), 30), labels=[0, 0, 0, 1], fontsize=12, linewidth=0.01, fmt=diag_config.formatParallelsDeg)
        # 填充颜色
        CS2 = m.contourf(x, y, reglist[i], color_arr, cmap=cm.get_cmap('own'),extend='both')
        # 添加色度条
        m.colorbar(CS2, location='bottom', pad="15%")
        # 绘制海岸线
        m.drawcoastlines(linewidth=0.2)
        ##设置标题
        plt.title(get_title(var,i), size=17)
        plt.savefig(path + "eof_slp_"+str(i+1)+".png",  dpi=200)
        plt.close()
        print("ok")


# 本地服务：获取大气环流无层次数据，返回大气环流的无层次年际平均值数组，并且返回绘图所需的经纬度
# 参数：var： 环流变量，示例：hgt:位势高度，vwnd:经向风，uwnd:纬向风，slp:海平面气压，wind:矢量风，air:大气温度
#       start_date：  起始时间，格式：yyyyMMdd
#       end_date：  结束时间，格式：yyyyMMdd
#       t_year：年份差，格式：整型
#       slat：起始纬度，范围：-90~90
#       elat：终止纬度，范围：-90~90
#       slon：起始经度，范围：0~360
#       elon：终止经度，范围：0~360
#       path：nc文件根路径
#       pattern：计算方式，0：得到平均场序列[返回二维数组]，1：得到时间平均值序列[返回一维数组]
def getSlpList(var,start_date,end_date,t_year,slat,elat,slon,elon,path,pattern):
    st = algorithm.get_day_of_year(start_date)
    ed = algorithm.get_day_of_year(end_date)
    start_year = int(start_date[0:4])  # 获取开始年份
    necp_slon, necp_slat = algorithm.xy_ncep_convert(slon, slat)
    necp_elon, necp_elat = algorithm.xy_ncep_convert(elon, elat)
    # 读取每年的位势高度数据并计算平均高度
    resultList = []
    for year in range(t_year):
        filepath = path+"\\" + var + "\\" + var + "." + str(start_year) + ".nc"
        # 读取数据
        nc_obj = nc.Dataset(filepath)
        lat = nc_obj.variables['lat'][necp_slat:necp_elat+1].squeeze()
        lon = nc_obj.variables['lon'][necp_slon:necp_elon+1].squeeze()
        atmos = nc_obj.variables[var][st:ed, necp_slat:necp_elat+1, necp_slon:necp_elon+1].squeeze()
        if pattern==0:
            ele_ave = np.mean(atmos, 0)
        if pattern==1:
            ele_ave = atmos.mean()
        resultList.append(ele_ave)
        start_year = start_year + 1
    return np.ma.masked_array(resultList),lat,lon





if __name__ == '__main__':
    # - - - - - - - - - - - 参数区域- - - - - - - - - - -
    var = "slp" #要素
    start = "19900601"  # 开始时间
    end = "20100801"  # 结束时间
    slat = -88  # 起始纬度
    elat = 70  # 终止纬度
    slon = 0  # 起始经度
    elon = 360  # 终止经度
    num = 5 #模态个数
    path = "D:\\bcacfs\\nc\\atmos"  # 环流NC文件路径
    pattern = 0
    qua = 1 #1：距平，2：标准化，3：原始值
    out_path = "D:\\"
    # - - - - - - - - - - - 计算区域 - - - - - - - - - - - - -
    t_year = algorithm.get_year_result(start, end)  # 获取年份差
    seqa,lat,lon = getSlpList(var,start ,end ,t_year,slat,elat,slon,elon,path,pattern)#获取时间段内大气环流原始值序列,以及需要绘图所需经纬度范围
    seq = algorithm.get_result_list(seqa,qua,t_year)
    eofs,pcs = algorithm.eof(seq,num)


    # - - - - - - - - - - - 绘图区域 - - - - - - - - - - - - - -
    draw_map(var,start,end,eofs,lat,lon,out_path)

