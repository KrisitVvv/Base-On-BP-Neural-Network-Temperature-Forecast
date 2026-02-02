import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from datetime import datetime
import matplotlib.dates as mdates
import os
from matplotlib.font_manager import FontProperties

parser = argparse.ArgumentParser(description='Render weather prediction results from forecast.xlsx')
parser.add_argument('-r', '--read-path', type=str, default='./forecast', 
                    help='read file path，default:./forecast')
args = parser.parse_args()

file_path = os.path.join(args.read_path, 'forecast.xlsx')
df = pd.read_excel(file_path)
time = df['日期']
max_temp = df['预测最大温度']
min_temp = df['预测最小温度']
font = FontProperties(fname='msyh.ttc')
fig, ax = plt.subplots()
date_format = mdates.DateFormatter('2025-%m')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(date_format)
ax.plot(time, max_temp, color='red', label='Max Temperature')
ax.plot(time, min_temp, color='blue', label='Min Temperature')
ax.set_xlabel('Date', fontproperties=font)
ax.set_ylabel('Temperature(℃)', fontproperties=font)
ax.set_title('Forecast Temperature', fontproperties=font)
ax.legend(prop=font)
img_path = './output/predicate.jpg'
img_dir = os.path.dirname(img_path)
os.makedirs(img_dir, exist_ok=True) 
plt.xticks(rotation=45)
plt.savefig(img_path,bbox_inches='tight',dpi=500)
plt.show()