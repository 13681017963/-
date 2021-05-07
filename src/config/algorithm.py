"""
算法
"""
import numpy as np
from src.config import station_config
from eofs.standard import Eof
from datetime import datetime
from numpy import linalg as la
from scipy.interpolate import griddata
import time
# from numba import jit, njit
# import numba

#计算距平或标准化
def get_result_list(list,qua,t_year):
    ave = sum(list) / t_year
    if (qua == 1):
        resultArr = get_ano(list, ave)
    if (qua == 2):
        resultArr = get_standard(list, ave)
    if (qua == 3):
        resultArr = np.ma.masked_array(list)
    return np.ma.masked_array(resultArr)

# 计算每年的要素场距平,返回MaskedArray
def get_ano(resultList,ave):
     anoList = []
     for result in range(len(resultList)):
        ele = resultList[result]
        ano = ele - ave
        anoList.append(ano)
     anoArr = np.ma.masked_array(anoList)
     print("距平:"+str(anoList))
     return anoArr

# 计算每年的要素场标准化,返回MaskedArray
def get_standard(resultList,ave):
     strList = []
     std = np.std(resultList)
     for result in range(len(resultList)):
         ele = resultList[result]
         m_ele = (ele - ave)/std
         strList.append(m_ele)
     strArr = np.ma.masked_array(strList)
     print("标准化:"+str(strList))
     return strArr


# 获取时间连续的月份差，格式yyyyMM
def get_month_continuous_result(str, etr):
    v_year_end = datetime.strptime(etr, '%Y%m').year
    v_month_end = datetime.strptime(etr, '%Y%m').month
    v_year_start = datetime.strptime(str, '%Y%m').year
    v_month_start = datetime.strptime(str, '%Y%m').month
    interval = (v_year_end - v_year_start)*12 + (v_month_end - v_month_start) + 1
    return interval


# 获取年际变化的月份差，格式yyyyMM
def get_month_result(str,etr):
    date_s = int(str[4:6])  # 获取开始月份
    date_e = int(etr[4:6])  # 获取结束月份
    if date_s <= date_e:
        month = date_e - date_s + 1
    if date_s > date_e:
        month = (12 - date_s + 1)+date_e
    return month

# 获取选择的年份差，格式yyyyMM
def get_year_result(str,etr):
    date_s = int(str[0:4])  # 获取开始年份
    date_e = int(etr[0:4])  # 获取结束年份
    year = date_e - date_s + 1 # 包含开始及结束年份
    return year

# 获取NOAA海温资料的月份，格式yyyyMM
def get_sst_month(str):
    date_p = datetime.strptime(str, '%Y%m').date()
    month = (date_p.year-1854)*12+date_p.month-1
    return month


# 获取NOAA射出长波辐射月资料的月份，格式yyyyMM
def get_olr_month(str):
    date_p = datetime.strptime(str, '%Y%m').date()
    month = (date_p.year-1979)*12+date_p.month-1
    return month

# 获取NOAA射出长波辐射日资料的日份，格式yyyyMMdd
def get_olr_day(str):
    y = int(str[0:4])  # 获取年份
    m = int(str[4:6])  # 获取月份
    d = int(str[6:8])  # 获取“日”
    dt = datetime(y, m, d)
    time = int(dt.strftime('%j')) - 1
    date_p = datetime.strptime(str, '%Y%m%d').date()
    month = (date_p.year-1979)*365+time
    return month

# 经纬度转换为NOAA资料经纬度,180*89
def xy_noaa_convert(nx,ny):
    x = nx/2
    y = ny/2 + 45
    return int(x),int(y)

# 经纬度转换为NOAA资料经纬度,360*180
def xy_noaa_convert2(nx,ny):
    x = nx/2
    y = ny/2 + 90
    return int(x),int(y)

#读取制作res站点数据
def make_station_list(sta_order_list,list):
    res_list = []
    for id in range(len(sta_order_list)):
        res = np.zeros(4, dtype=np.float)
        station = sta_order_list[id]
        latlon = station_config.get_station_latlon(station)
        latlon_arr = latlon.split(',')
        data = list[id]
        res[0] = 0
        res[1] = latlon_arr[0]
        res[2] = latlon_arr[1]
        res[3] = data
        res_list.append(res)
    return np.array(res_list)

# 经纬度转换为NCEP资料经纬度,144*73
def xy_ncep_convert(nx,ny):
    x = nx / 2.5
    if ny > 0 and ny <=90:
        y = ny / 2.5 + 90 / 2.5
    elif ny<0 and ny>=-90:
        y = ny / 2.5 + 90 / 2.5
    else:
        y = ny / 2.5
    return int(x),int(y)

#获取某日期是一年中的第几天
def get_day_of_year(date):
    y = int(date[0:4])  # 获取年份
    m = int(date[4:6])  # 获取月份
    d = int(date[6:8])  # 获取“日”
    dt = datetime(y, m, d)
    time = int(dt.strftime('%j'))-1
    return time

#获取指数集资料行号，日期格式yyyyMM
def get_ind_num(date):
    year = int(date[0:4])
    month = int(date[4:6])
    if(year>=1952):
        month = (year -1951)*12 + month
    return month

#EOF分析计算
def eof(seq,num):
    mask = seq.mask
    solver = Eof(seq)
    eof_list = np.ma.array(solver.eofs(), mask=mask).filled(-9.96921E+36)
    pcs_list = solver.pcs()
    print("EOF空间模态个数" + str(len(eof_list)))
    if num==0:
        return eof_list,pcs_list
    else:
        return eof_list[0:num - 1], pcs_list[0:num - 1]


#相关分析计算
def corr(seql, seqr):
    datanum_e = seql.shape[1]
    res = np.corrcoef(seql.T, seqr)[:datanum_e, datanum_e]
    print("相关系数："+str(res))
    return res

#回归分析计算
def reg(seqe, seqa):
    res = np.polyfit(seqa, seqe, deg=1)[0]
    print("回归系数：" + str(res))
    return res


#进行左右场SVD矩阵计算,返回左场空间场，左场时间序列，右场空间场，右场时间序列
def svd_ele_oce(seql,ori_seqr,t_year):
    datanuml = seql.shape[0]  #左场序列有效点
    (t, m, n) = ori_seqr.shape;
    seql = np.array(seql)
    seqr = np.array(ori_seqr.reshape(t, m * n).T)
    datanumr = seqr.shape[0] # 右场序列有效点
    if(datanuml<=datanumr):
        mtc = np.matmul(np.array(seql),np.transpose(seqr))
        mtc = mtc/t_year
        start = time.time()
        mtl,mte,mtr = la.svd(mtc)
        end = time.time()
        print(end - start)
        lt = np.matmul(np.transpose(mtl),seql)
        lt = lt/(np.math.sqrt(datanuml))
        ltime = lt[0:t_year,:]

        rt = np.matmul(np.transpose(mtr), seqr)
        rt = rt / (np.math.sqrt(datanumr))
        rtime = rt[0:t_year,:]
    if(datanuml>datanumr):
        mtc = np.matmul(seqr, np.transpose(seql))
        mtc = mtc / t_year
        mtl, mte, mtr = la.svd(mtc)

        lt = np.matmul(np.transpose(mtr), seql)
        lt = lt / (np.math.sqrt(datanuml))
        ltime = lt[1:t_year, :]

        rt = np.matmul(np.transpose(mtl), seqr)
        rt = rt / (np.math.sqrt(datanumr))
        rtime = rt[1:t_year, :]

    lhomo = np.corrcoef(seql, ltime, bias=True)[:datanuml, datanuml:]
    rhomo = np.transpose(np.corrcoef(seqr, rtime, bias=True)[:datanumr, datanumr:].reshape(m, n, t), axes=(2, 0, 1))
    return lhomo,ltime,rhomo,rtime




def get_svd_result(mtc):
    mtl, mte, mtr = la.svd(mtc,full_matrices=1)
    return mtl,mte,mtr

#进行环流与海温计算SVD,左右场SVD矩阵计算,返回左场空间场，左场时间序列，右场空间场，右场时间序列
def svd_atmo_oce(seql,ori_seqr,t_year):
    (a, b, c) = seql.shape;
    seql = np.array(seql.reshape(a, b * c).T)
    datanuml = seql.shape[0]  # 左场序列有效点

    (t, m, n) = ori_seqr.shape;
    seqr = np.array(ori_seqr.reshape(t, m * n).T)
    datanumr = seqr.shape[0] # 右场序列有效点
    if(datanuml<=datanumr):
        mtc = np.matmul(np.array(seql),np.transpose(seqr))
        mtc = mtc/t_year
        start = time.time()
        mtl,mte,mtr = get_svd_result(mtc)
        end = time.time()
        print(end - start)
        lt = np.matmul(np.transpose(mtl),seql)
        lt = lt/(np.math.sqrt(datanuml))
        ltime = lt[0:t_year,:]

        rt = np.matmul(np.transpose(mtr), seqr)
        rt = rt / (np.math.sqrt(datanumr))
        rtime = rt[0:t_year,:]
    if(datanuml>datanumr):
        mtc = np.matmul(seqr, np.transpose(seql))
        mtc = mtc / t_year
        mtl, mte, mtr = get_svd_result(mtc)

        lt = np.matmul(np.transpose(mtr), seql)
        lt = lt / (np.math.sqrt(datanuml))
        ltime = lt[1:t_year, :]

        rt = np.matmul(np.transpose(mtl), seqr)
        rt = rt / (np.math.sqrt(datanumr))
        rtime = rt[1:t_year, :]

    #lhomo = np.corrcoef(seql, ltime, bias=True)[:datanuml, datanuml:]
    lhomo = np.transpose(np.corrcoef(seql, ltime, bias=True)[:datanuml, datanuml:].reshape(b, c, a), axes=(2, 0, 1))
    rhomo = np.transpose(np.corrcoef(seqr, rtime, bias=True)[:datanumr, datanumr:].reshape(m, n, t), axes=(2, 0, 1))
    return lhomo,ltime,rhomo,rtime


#shape图形绘制
def get_RegionID_by_XML(XmlFileName,RegionID_IN):
    from xml.dom import minidom
    xmldoc = minidom.parse(XmlFileName)
    AllObjects  =  xmldoc.getElementsByTagName("Object")
    for  Object  in  AllObjects:
        RegionID = Object.getElementsByTagName("RegionID")[0].childNodes[0].nodeValue
        if(RegionID_IN!=RegionID):
            continue
        dict1={}
        INITDIR = Object.getElementsByTagName("INITDIR")[0].childNodes[0].nodeValue
        RegionName = Object.getElementsByTagName("RegionName")[0].childNodes[0].nodeValue
        RegionShapeFile= Object.getElementsByTagName("RegionShapeFile")[0].childNodes[0].nodeValue
        ProjType= Object.getElementsByTagName("ProjType")[0].childNodes[0].nodeValue
        DrawArea= Object.getElementsByTagName("DrawArea")[0].childNodes[0].nodeValue
        RegionArea= Object.getElementsByTagName("RegionArea")[0].childNodes[0].nodeValue
        I_STA_TYPE= Object.getElementsByTagName("I_STA_TYPE")[0].childNodes[0].nodeValue

        LongitudeInfo= Object.getElementsByTagName("LongitudeInfo")[0].childNodes[0].nodeValue
        LatitudeInfo= Object.getElementsByTagName("LatitudeInfo")[0].childNodes[0].nodeValue
        StationInfoFile= Object.getElementsByTagName("StationInfoFile")[0].childNodes[0].nodeValue
        InterpToFile = Object.getElementsByTagName("InterpToFile")[0].childNodes[0].nodeValue

        Desc= Object.getElementsByTagName("Desc")[0].childNodes[0].nodeValue
        dict1['RegionID']=RegionID
        dict1['INITDIR']=INITDIR
        dict1['RegionName']=RegionName
        dict1['RegionShapeFile']=RegionShapeFile
        dict1['ProjType']=ProjType
        dict1['DrawArea']=DrawArea
        dict1['RegionArea']=RegionArea
        dict1['I_STA_TYPE']=I_STA_TYPE
        dict1['LongitudeInfo']=LongitudeInfo
        dict1['LatitudeInfo']=LatitudeInfo
        dict1['StationInfoFile']=StationInfoFile
        dict1['InterpToFile']=InterpToFile
        dict1['Desc']=Desc

        if([] == Object.getElementsByTagName("ShapeFiles")):
            continue
        ShapeFiles  =  Object.getElementsByTagName("ShapeFiles")[0]
        FFF = ShapeFiles.getElementsByTagName("F")
        list1=[]
        for F1 in FFF:
            dict2={}
            Shapefile = F1.getElementsByTagName("ShapeFile")[0].childNodes[0].nodeValue
            COLOR = F1.getElementsByTagName("COLOR")[0].childNodes[0].nodeValue
            LineWidth = F1.getElementsByTagName("LineWidth")[0].childNodes[0].nodeValue
            dict2['Shapefile']=Shapefile
            dict2['COLOR']=COLOR
            dict2['LineWidth']=LineWidth
            list1.append(dict2)
            dict1['Shapefiles']=list1
        return dict1

    '''
    func : 将站点数据插值到等经纬度格点
    inputs:
        lon: 站点的经度
        lat: 站点的纬度
        data: 对应经纬度站点的 气象要素值
        loc_range: [lat_min,lat_max,lon_min,lon_max]。站点数据插值到loc_range这个范围
        det_grid: 插值形成的网格空间分辨率
        method: 所选插值方法，默认 0.125
    return:

        [lon_grid,lat_grid,data_grid]
    '''
def interp2d_station_to_grid(lon, lat, data, loc_range=[18, 54, 73, 135],
                             det_grid=1, method='cubic'):

    # step1: 先将 lon,lat,data转换成 n*1 的array数组
    lon = np.array(lon).reshape(-1, 1)
    lat = np.array(lat).reshape(-1, 1)
    data = np.array(data).reshape(-1, 1)

    # shape = [n,2]
    points = np.concatenate([lon, lat], axis=1)

    # step2:确定插值区域的经纬度网格
    lat_min = loc_range[0]
    lat_max = loc_range[1]
    lon_min = loc_range[2]
    lon_max = loc_range[3]

    lon_grid, lat_grid = np.meshgrid(np.arange(lon_min, lon_max + det_grid, det_grid),
                                     np.arange(lat_min, lat_max + det_grid, det_grid))

    # step3:进行网格插值
    grid_data = griddata(points, data, (lon_grid, lat_grid), method=method)
    grid_data = grid_data[:, :, 0]

    # 保证纬度从上到下是递减的
    if lat_grid[0, 0] < lat_grid[1, 0]:
        lat_grid = lat_grid[-1::-1]
        grid_data = grid_data[-1::-1]

    return [lon_grid, lat_grid, grid_data]



    '''
    func: 将等经纬度网格值 插值到 离散站点。使用griddata进行插值
    inputs: 
        all_data,形式为：[grid_lon,grid_lat,data] 即[经度网格，纬度网格，数值网格]
        station_lon: 站点经度
        station_lat: 站点纬度。可以是 单个点，列表或者一维数组
        method: 插值方法,默认使用 cubic
    '''
def grid_interp_to_station(all_data, station_lon, station_lat, method='cubic'):
    station_lon = np.array(station_lon).reshape(-1, 1)
    station_lat = np.array(station_lat).reshape(-1, 1)

    lon = all_data[0].reshape(-1, 1)
    lat = all_data[1].reshape(-1, 1)
    data = all_data[2].reshape(-1, 1)

    points = np.concatenate([lon, lat], axis=1)

    station_value = griddata(points, data, (station_lon, station_lat), method=method)

    station_value = station_value[:, :, 0]

    return station_value




    '''
    func:获取与给定经纬度值的点最近的等经纬度格点的经纬度index
    inputs:
        point_lon_lat: 给定点的经纬度，eg:[42.353,110.137]
        lon_grid: 经度网格
        lat_grid: 纬度网格
    return:
        index: [index_lat,index_lon]
    '''
def get_nearest_point_index(point_lon_lat, lon_grid, lat_grid):

    # step1: 获取网格空间分辨率;默认纬度和经度分辨率一致
    det = lon_grid[0, 1] - lon_grid[0, 0]

    # step2:
    point_lon = point_lon_lat[0]
    point_lat = point_lon_lat[1]

    lon_min = np.min(lon_grid)
    lat_min = np.min(lat_grid)
    #    lat_max = np.max(lat_grid)

    index_lat = round((point_lat - lat_min) / det)
    index_lon = round((point_lon - lon_min) / det)

    # 由于默认的 lat_max值对应的index为0，因此需要反序
    index_lat = lat_grid.shape[0] - index_lat - 1

    return [int(index_lat), int(index_lon)]
