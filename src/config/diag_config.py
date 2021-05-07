"""
大气环流查询字典
"""
import numpy as np
import matplotlib.colors as col
# 获取层次的资料值
def get_level(lev):
     my_dict = {"1000": 0, "925": 1, "850": 2, "700": 3, "600": 4, "500": 5, "400": 6, "300": 7, "250": 8,
                "200": 9, "150": 10, "100": 11, "70": 12, "50": 13, "30": 14, "20": 15, "10": 16}
     value = my_dict[lev]
     return value

# 获取气候监测诊断分析要素名称
def get_diag_name(var):
     my_dict = {"tem":"气温","pre":"降水","hgt": "位势高度场", "vwnd": "经向风场", "uwnd": "纬向风场", "slp": "海平面气压场", "air": "大气温度场", "wind": "矢量风场","index":"指数"}
     value = my_dict[var]
     return value

# 获取气候监测诊断分析要素名称
def get_index_name(var):
     my_dict = {'1': '北半球副高面积指数','2': '北非副高面积指数','3': '北非-大西洋-北美副高面积指数','4': '印度副高面积指数'}
     value = my_dict[var]
     return value


# 获取相关分析色系表与取值范围
def get_corr_cmap_arr():
     cpool = ['#660bff', '#3333ff', '#3966ff', '#6699ff', '#ccecfe', '#ffffff',
              '#feffcc', '#f4cc64', '#ed972f', '#e96429', '#e62f2d']
     cmap1 = col.LinearSegmentedColormap.from_list('own', cpool[0:11])
     arr =  np.array([-0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4])
     return cmap1,arr

# 获取回归分析色系表与取值范围
def get_reg_cmap_arr(var):
     cpool = ['#660bff', '#3333ff', '#3966ff', '#6699ff', '#ccecfe', '#ffffff',
              '#feffcc', '#f4cc64', '#ed972f', '#e96429', '#e62f2d']
     cmap1 = col.LinearSegmentedColormap.from_list('own', cpool[0:11])
     if var == "hgt":
          arr =  np.array([-40, -30, -20, -10, -2, 2, 10, 20, 30, 40])
     if var == "vwnd":
          arr = np.array([-40, -30, -20, -10, -2, 2, 10, 20, 30, 40])
     if var == "uwnd":
          arr = np.array([-40, -30, -20, -10, -2, 2, 10, 20, 30, 40])
     if var == "air":
          arr = np.array([-40, -30, -20, -10, -2, 2, 10, 20, 30, 40])
     if var == "ind":
          arr =  np.array([-0.4, -0.3, -0.2, -0.1, -0.02, 0.02, 0.1, 0.2, 0.3, 0.4])
     return cmap1,arr

# 获取EOF色系表与取值范围
def get_eof_cmap_arr():
     cpool = ['#660bff', '#3333ff', '#3966ff', '#6699ff', '#ccecfe', '#ffffff',
              '#feffcc', '#f4cc64', '#ed972f', '#e96429', '#e62f2d']
     cmap1 = col.LinearSegmentedColormap.from_list('own', cpool[0:11])
     arr =  np.array([-0.04, -0.03, -0.02, -0.01, -0.002, 0.002, 0.01, 0.02, 0.03, 0.04])
     return cmap1,arr

# 获取海温距平色系表与取值范围
def get_ano_oce_cmap_arr():
     cpool = ['#1900d8', '#212ec5', '#2c37f9', '#4f9edd', '#b5f9f6',
              '#f8ffc1', '#E6DD31', '#e6b02d','#eb8128', '#e62f2d']
     cmap1 = col.LinearSegmentedColormap.from_list('own', cpool[0:10])
     arr =  np.array([ -4, -3, -2, -1, -0.5, 0, 0.5, 1,  2,  3, 4], dtype=np.float)
     return cmap1,arr

# 获取海温原始色系表与取值范围
def get_ori_oce_cmap_arr():
     cpool = ['#7f12ff', '#710fff', '#6519ff', '#5926ff', '#4c32ff', '#3f3fff', '#3f65ff', '#498cff', '#5ab3fe', '#6dd9fd', '#7efcfc', '#7dfcda', '#7cfcb3',
              '#7bfc8d', '#7afc67', '#7afc43', '#7afc36', '#8bfc27', '#b1fd1a', '#d9fe0e', '#feff00', '#f8e500', '#f4cc07', '#f0b118', '#ed9720', '#eb7d26',
              '#e96429', '#e7482c', '#e62f2d', '#e82725', '#e8211f', '#ef1513', '#cd2827', '#b22621', '#993219']
     cmap1 = col.LinearSegmentedColormap.from_list('own', cpool[0:35])
     arr =  np.array([-2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32], dtype=np.int)
     return cmap1,arr


# 获取射出长波辐射距平色系表与取值范围
def get_ano_olr_cmap_arr():
     cpool = [ '#3661f1', '#4f9edd', '#62c5eb', '#ffffff', '#e6dd33', '#e0ac2c', '#eb7f28']
     cmap1 = col.LinearSegmentedColormap.from_list('own', cpool[0:8])
     arr =  np.array([-45, -30, -15,  15, 30, 45], dtype=np.int)
     return cmap1,arr

# 获取射出长波辐射原始色系表与取值范围
def get_ori_olr_cmap_arr():
     cpool = ['#801fef', '#1d2bb1', '#293df9', '#3661f1', '#74a2f8', '#5ab4f2', '#e6dd33', '#eb8128', '#e62f2d', '#9d1d1d']
     cmap1 = col.LinearSegmentedColormap.from_list('own', cpool[0:10])
     arr =  np.array([160, 180, 200, 220, 240, 260, 280, 300, 320], dtype=np.int)
     return cmap1,arr

# 获取大气环流图形标题
def get_ano_atmos_title(var,lev):
     if var == "hgt":
          result = "NCEP/NCAR "+str(lev)+"hPa位势高度平均场及距平分布图"
     if var=="vwnd":
          result = "NCEP/NCAR "+str(lev)+"hPa经向风平均场及距平分布图"
     if var=="uwnd":
          result = "NCEP/NCAR "+str(lev)+"hPa纬向风平均场及距平分布图"
     if var=="slp":
          result = "NCEP/NCAR 海平面气压平均场及距平分布图"
     return result

# 获取位势高度色系表与取值范围
def get_ano_atmos_cmap_arr(var):
     #位势高度色系表与取值范围
     if var=="hgt":
          cpool = ['#3564d3', '#53a5f5', '#77b9fa', '#96d2fa', '#b4f0fb', '#ffffff', '#f9e878',
                   '#f2bf39', '#ee9f1e', '#e85d2a', '#e62f2d']
          cmap1 = col.LinearSegmentedColormap.from_list('own', cpool[0:11])
          arr = np.array([-80, -60, -40, -20, -10, -5, 5, 10, 20, 40, 60, 80], dtype=np.int)
     #海平面气压色系表与取值范围
     if var=="slp":
          cpool = ['#3a6eeb', '#4382f0', '#4c96f5', '#53a5f5', '#77b9fa', '#96d2fa', '#b4f0fb', '#f9e878',
                   '#f2bf39', '#ee9f1e', '#e85d2a', '#e62f2d', '#e22d2c', '#c12524']
          cmap1 = col.LinearSegmentedColormap.from_list('own', cpool[0:14])
          arr = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int)
     # 经向风色系表与取值范围
     if var=="vwnd":
          cpool = ['#a11cc8', '#8214dc', '#2a3dff', '#62c8c9', '#66d28d', '#a0e632', '#e6dd31', '#eb8127',
                   '#e73838', '#e72e81']
          cmap1 = col.LinearSegmentedColormap.from_list('own', cpool[0:10])
          arr = np.array([-12, -9, -6, -3, 0, 3, 6, 9, 12], dtype=np.int)
     # 纬向风色系表与取值范围
     if var == "uwnd":
          cpool = ['#a11cc8', '#8214dc', '#2a3dff', '#62c8c9', '#66d28d', '#a0e632', '#e6dd31', '#eb8127',
                   '#e73838', '#e72e81']
          cmap1 = col.LinearSegmentedColormap.from_list('own', cpool[0:10])
          arr = np.array([-20, -15, -10, -5,  0, 5, 10, 15, 20], dtype=np.int)

     return cmap1,arr

# 自定义纬度格式
def formatParallelsDeg(deg):
    if deg > 0:
        return str(deg) + "N"
    if deg < 0:
        return str(abs(deg)) + "S"
    if deg == 0:
        return "EQ"
# 自定义经度格式
def formatMeridiansDeg(deg):
    if deg > 180:
        return str(np.around(180-(deg-180), 1)) + "W"
    if deg < 180:
        return str(np.around(abs(deg), 1)) + "E"
    if deg == 180:
        return "180"
