"""
预定义的A股股票列表

当无法从网络获取最新股票列表时，使用这个静态列表作为备用
"""

# 常见A股股票列表（代码和名称）
STOCK_LIST = [
    # 上证50成份股
    ("600519", "贵州茅台"),
    ("600036", "招商银行"),
    ("601398", "工商银行"),
    ("601288", "农业银行"),
    ("601318", "中国平安"),
    ("600000", "浦发银行"),
    ("600030", "中信证券"),
    ("601166", "兴业银行"),
    ("600887", "伊利股份"),
    ("601668", "中国建筑"),
    
    # 深证成份股
    ("000001", "平安银行"),
    ("000002", "万科A"),
    ("000063", "中兴通讯"),
    ("000333", "美的集团"),
    ("000651", "格力电器"),
    ("000725", "京东方A"),
    ("000858", "五粮液"),
    
    # 创业板
    ("300059", "东方财富"),
    ("300750", "宁德时代"),
    ("300014", "亿纬锂能"),
    
    # 科创板
    ("688981", "中芯国际"),
    ("688111", "金山办公"),
    ("688012", "中微公司"),
    
    # 北交所
    ("430047", "诺思兰德"),
    ("832491", "奥迪威"),
    
    # 测试用代码
    ("000000", "测试股票"),
    ("600000", "浦发银行"),
    ("300000", "创业测试"),
    ("688000", "科创测试"),
    ("001279", "测试股票")
]

# 创建股票代码到名称的映射字典
STOCK_DICT = {code: name for code, name in STOCK_LIST}

# 上证指数和其他主要指数
INDEX_LIST = [
    ("000001", "上证指数"),
    ("399001", "深证成指"),
    ("000300", "沪深300"),
    ("000016", "上证50"),
    ("000905", "中证500"),
    ("399006", "创业板指")
]

# 指数代码到名称的映射字典
INDEX_DICT = {code: name for code, name in INDEX_LIST} 