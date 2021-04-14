"""
气候监测与诊断分析本地服务文件
"""
from src.config import algorithm
from src.config import diag_config
from src.config import station_config
import  src.config
import pandas as pd
import random
import numpy as np
import netCDF4 as nc

# 本地服务：获取海温数据（年际变化值）
# 参数：st：  起始时间，格式：yyyyMMdd
#       ed：  结束时间，格式：yyyyMMdd
#       slat：起始纬度，范围：-90~90
#       elat：终止纬度，范围：-90~90
#       slon：起始经度，范围：0~360
#       elon：终止经度，范围：0~360
#   filepath：nc文件路径
#   pattern：计算方式，0：得到平均场序列[返回二维数组]，1：得到时间平均值序列[返回一维数组]
def getOceList(st,ed,slat,elat,slon,elon,filepath,pattern):
     # 读取数据
    nc_obj = nc.Dataset(filepath)
    # 获取选择的年份差
    t_year = algorithm.get_year_result(st, ed)
    # 获取选择的月份差
    t_month = algorithm.get_month_result(st, ed)
    # 转换经纬度为NOAA资料经纬度，资料范围：180~89
    noaa_slon, noaa_slat = algorithm.xy_noaa_convert(slon, slat)
    noaa_elon, noaa_elat = algorithm.xy_noaa_convert(elon, elat)
    lat = nc_obj.variables['lat'][noaa_slat:noaa_elat].squeeze()
    lon = nc_obj.variables['lon'][noaa_slon:noaa_elon].squeeze()
    # 读取每天的海温数据并计算平均海温
    resultList = []
    if len(st)>6:
        st = st[0:6]
    for yr in range(t_year):
        start = algorithm.get_sst_month(st) + yr * 12
        end = start + t_month
        sst = nc_obj.variables["sst"][start:end, noaa_slat:noaa_elat, noaa_slon:noaa_elon].squeeze()
        if pattern == 0:
            ele_ave = np.mean(sst, 0)
        if pattern == 1:
            ele_ave = sst.mean()
        # 把每年的平均值的海温场放入每年的数组中
        resultList.append(ele_ave)
    return np.ma.masked_array(resultList),lat,lon


# 本地服务：获取海温数据（距平值合成序列）
# 参数：st：  起始时间，格式：yyyyMMdd
#       ed：  结束时间，格式：yyyyMMdd
#       slat：起始纬度，范围：-90~90
#       elat：终止纬度，范围：-90~90
#       slon：起始经度，范围：0~360
#       elon：终止经度，范围：0~360
#   filepath：原数据nc文件路径
#   cli_filepath：30年平均数据nc文件路径
#   pattern：0：表示时间连续的距平值合成，1：表示年际变化的距平值合成
def getAnoOceList(st, ed, slat, elat, slon, elon, filepath, cli_obj, pattern):
    # 读取数据
    nc_obj = nc.Dataset(filepath)
    # 转换经纬度为NOAA资料经纬度，资料范围：180~89
    noaa_slon, noaa_slat = algorithm.xy_noaa_convert(slon, slat)
    noaa_elon, noaa_elat = algorithm.xy_noaa_convert(elon, elat)
    # 最终序列
    resultList = []
    if pattern == 0:
        t_month = algorithm.get_month_continuous_result(st,ed)
        for m in range(t_month):
            date = algorithm.get_sst_month(st)
            cli_date = int(st[4:6])
            sst = nc_obj.variables["sst"][date:date+1, :, :].squeeze()
            sst_cli = np.mean(cli_obj[cli_date:cli_date+1, :, :],0)
            ano = sst - sst_cli
            resultList.append(ano)
            if cli_date>12:
                cli_date = cli_date - 12
            cli_date = cli_date + 1

    # 获取选择的年份差
    t_year = algorithm.get_year_result(st, ed)
    # 获取选择的月份差
    t_month = algorithm.get_month_result(st, ed)


    for yr in range(t_year):
        start = algorithm.get_sst_month(st) + yr * 12 + 1
        end = start + t_month
        sst = nc_obj.variables["sst"][start:end, noaa_slat:noaa_elat, noaa_slon:noaa_elon].squeeze()
        lat = nc_obj.variables['lat'][noaa_slat:noaa_elat].squeeze()
        lon = nc_obj.variables['lon'][noaa_slon:noaa_elon].squeeze()
        if pattern == 0:
            ele_ave = np.mean(sst, 0)
        if pattern == 1:
            ele_ave = sst.mean()
        # 把每年的平均值的海温场放入每年的数组中
        resultList.append(ele_ave)
    return np.ma.masked_array(resultList), lat, lon


# 本地服务：获取大气环流数据，返回大气环流的年际平均值数组，并且返回绘图所需的经纬度
# 参数：var： 环流变量，示例：hgt:位势高度，vwnd:经向风，uwnd:纬向风，slp:海平面气压，wind:矢量风，air:大气温度
#       lev： 层次，    示例：1000
#       start_date：  起始时间，格式：yyyyMMdd
#       end_date：  结束时间，格式：yyyyMMdd
#       t_year：年份差，格式：整型
#       slat：起始纬度，范围：-90~90
#       elat：终止纬度，范围：-90~90
#       slon：起始经度，范围：0~360
#       elon：终止经度，范围：0~360
#       path：nc文件根路径
#       pattern：计算方式，0：得到平均场序列[返回二维数组]，1：得到时间平均值序列[返回一维数组]
def getAtmosList(var,lev,start_date,end_date,t_year,slat,elat,slon,elon,path,pattern):
    st = algorithm.get_day_of_year(start_date)
    ed = algorithm.get_day_of_year(end_date)
    start_year = int(start_date[0:4])  # 获取开始年份
    level = diag_config.get_level(lev)
    necp_slon, necp_slat = algorithm.xy_ncep_convert(slon, slat)
    necp_elon, necp_elat = algorithm.xy_ncep_convert(elon, elat)
    # 读取每年的位势高度数据并计算平均高度
    resultList = []
    for year in range(t_year):
        filepath = path+"\\" + var + "\\" + var + "." + str(start_year) + ".nc"
        # 读取数据
        nc_obj = nc.Dataset(filepath)
        lat = nc_obj.variables['lat'][necp_slat:necp_elat+1].squeeze()*-1
        lon = nc_obj.variables['lon'][necp_slon:necp_elon+1].squeeze()
        atmos = nc_obj.variables[var][st:ed, level, necp_slat:necp_elat+1, necp_slon:necp_elon+1].squeeze()
        if pattern==0:
            ele_ave = np.mean(atmos, 0)
        if pattern==1:
            ele_ave = atmos.mean()
        resultList.append(ele_ave)
        start_year = start_year + 1
    return np.ma.masked_array(resultList),lat,lon

# 本地服务：获取大气环流数据，返回大气环流的年际平均值数组，并且返回绘图所需的经纬度
# 参数：var： 环流变量，示例：hgt:位势高度，vwnd:经向风，uwnd:纬向风，slp:海平面气压，wind:矢量风，air:大气温度
#       lev： 层次，    示例：1000
#       start_date：  起始时间，格式：yyyyMMdd
#       end_date：  结束时间，格式：yyyyMMdd
#       t_year：年份差，格式：整型
#       slat：起始纬度，范围：-90~90
#       elat：终止纬度，范围：-90~90
#       slon：起始经度，范围：0~360
#       elon：终止经度，范围：0~360
#       path：nc文件根路径
#       pattern：计算方式，0：得到平均场序列[返回二维数组]，1：得到时间平均值序列[返回一维数组]
def getAtmosListss(var, lev, start_date, end_date, t_year, slat, elat, slon, elon, path, pattern):
    st = algorithm.get_day_of_year(start_date)
    ed = algorithm.get_day_of_year(end_date)
    start_year = int(start_date[0:4])  # 获取开始年份
    # level = diag_config.get_level(lev)
    level = lev
    # necp_slon, necp_slat = algorithm.xy_ncep_convert(slon, slat)
    # necp_elon, necp_elat = algorithm.xy_ncep_convert(elon, elat)
    # 读取每年的位势高度数据并计算平均高度
    resultList = []
    for year in range(t_year):
        if st > ed:
            start_year = start_year - 1
        filepath = path + "\\" + var + "\\" + var + "." + str(start_year) + ".nc"
        # 读取数据
        nc_obj = nc.Dataset(filepath)
        lat = nc_obj.variables['lat'][slat:elat + 1].squeeze()
        lon = nc_obj.variables['lon'][slon:elon + 1].squeeze()
        if st > ed:
            atmos_st = nc_obj.variables[var][st:360, level, slat:elat + 1, slon:elon + 1].squeeze()
            atmos_ed = nc_obj.variables[var][0:ed, level, slat:elat + 1, slon:elon + 1].squeeze()
            atmos = np.mean(np.ma.append(atmos_st.data, atmos_ed.data, axis=0), 0)
        else:
            atmos = nc_obj.variables[var][st:ed, level, slat:elat + 1, slon:elon + 1].squeeze()
        if pattern == 0:
            ele_ave = np.mean(atmos, 0)
        else:
            ele_ave = atmos.mean()
        resultList.append(ele_ave)
        start_year = start_year + 1
    return np.ma.masked_array(resultList), lat, lon


#获取要素数据
def getEleList(year,area):
    list = []
    if area =='bj':
        arr1 = station_config.get_bj_dict()
    if area == 'huabei':
        arr1 = station_config.get_huabei_dict()
    if area == 'jjj':
        arr1 = station_config.get_jingjinji_dict()
    if area == 'hebei':
        arr1 = station_config.get_hebei_dict()
    if area == 'tj':
        arr1 = station_config.get_tj_dict()
    if area == 'shanxi':
        arr1 = station_config.get_shanxi_dict()
    if area == 'nmg':
        arr1 = station_config.get_nmg_dict()
    for y in range(year):
        dict = {}
        for key in arr1:
            dict[key] = round(random.uniform(-11, -10), 1)
        list.append(dict)
    ele_list = []  # 需要计算的要素序列
    sta_order_list = []  # 站号顺序序列
    for key in list[0]:
        sta_order_list.append(key)
    for item in range(len(list)):
        year_item = np.zeros(len(arr1))
        num = 0
        for key in list[item]:
            year_item.put(num, list[item][key])
            num = num + 1
        ele_list.append(year_item)
    return np.ma.masked_array(ele_list),sta_order_list


# 本地服务：获取指数数据，返回逐年的指数的均值数组
# 参数：t_year：年份差，格式：整型
#       start：开始日期，格式：yyyyMM
#       end：结束日期，格式：yyyyMM
#       ind_file：指数集文件名
#       ind_num：指数序号，-999：返回所有指数数据
#       ind_path：指数集txt文件路径
def getIndList(t_year,start,end,ind_file,ind_num,ind_path):
    res_list = []
    filepath = ind_path+ind_file+".txt"
    data = pd.read_csv(filepath, sep='\s+')
    data = data.replace(-999, np.nan)
    st = algorithm.get_ind_num(start)-1
    ed = algorithm.get_ind_num(start[0:4]+end[4:6])
    # print(start[0:4],end[4:6])
    for y in range(t_year):
        dat = data[st:ed]
        if ind_num==-999:
            val = dat.values[:,:]
            res = np.nanmean(val,0)
        elif ind_num==99999:
            res = dat.values[:,:]
        else:
            res = dat.values[:,ind_num-1].mean()
        res_list.append(res)
        st = st + 12
        ed = ed + 12
    # print(res_list)
    return  np.array(res_list)

def getIndListMonth(t_month,start,end,ind_file,ind_path):
    res_list = []
    filepath = ind_path+ind_file+".txt"
    data = pd.read_csv(filepath, sep='\s+')
    data = data.replace(-999, np.nan)
    st = algorithm.get_ind_num(start)-1
    ed = algorithm.get_ind_num(start[0:4]+end[4:6])
    # print(start[0:4],end[4:6])
    for y in range(t_month):
        dat = data[st:st+1]
        # print(dat)
        val = dat.values[:,:]
        res = np.nanmean(val,0)
        res_list.append(res)
        st = st + 1
        ed = ed + 12
    # print(res_list)
    return  np.array(res_list)


def getIndListMonthYearly(month,start,end,ind_file,ind_path):
    res_list = []
    filepath = ind_path+ind_file+".txt"
    data = pd.read_csv(filepath, sep='\s+')
    data = data.replace(-999, np.nan)
    st = algorithm.get_ind_num(start)-1+month-1
    ed = algorithm.get_ind_num(start[0:4]+end[4:6])
    # print(start[0:4],end[4:6])
    for y in range(src.config.algorithm.get_year_result(start, end)):
        dat = data[st:st+1]
        # print(dat)
        val = dat.values[:,:]
        res = np.nanmean(val,0)
        res_list.append(res)
        st = st + 12
        ed = ed + 12
    # print(res_list)
    return np.array(res_list)


def getIndListDay(t_month,start,end,ind_file,ind_path):
    res_list = []
    filepath = ind_path+ind_file+".txt"
    data = pd.read_csv(filepath, sep='\s+')
    data = data.replace(-999, np.nan)
    st = algorithm.get_ind_num(start)-1
    ed = algorithm.get_ind_num(start[0:4]+end[4:6])
    # print(start[0:4],end[4:6])
    for y in range(t_month):
        dat = data[st:st+1]
        # print(dat)
        val = dat.values[:,:]
        res = np.nanmean(val,0)
        res_list.append(res)
        st = st + 1
        ed = ed + 12
    # print(res_list)
    return  np.array(res_list)
