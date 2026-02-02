import argparse
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from wordcloud import WordCloud
from nltk import FreqDist
import os

parser = argparse.ArgumentParser(description='Generate word cloud from Whether.xlsx')
parser.add_argument('-r', '--read-path', type=str, default='./reptile/output', 
                    help='read file path，default:./reptile/output')
args = parser.parse_args()

file_path = os.path.join(args.read_path, 'Whether.xlsx')
df = pd.read_excel(file_path)
column_3_data = df.iloc[:, 3]
text = ' '.join(column_3_data.astype('str'))
tokens = text.split()
freq_dist = FreqDist(tokens)
freq_dict = dict(freq_dist.items())
'''
# 统计词频
for word, frequency in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(word, frequency)
'''
font_path = 'msyh.ttc'
mask=cv2.imread('circle.jpg')
wc = WordCloud(background_color='white', max_words=40, font_path=font_path,mask=mask,margin=15)
wordcloud = wc.generate_from_frequencies(freq_dict)
plt.figure(dpi=300)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') 
plt.xticks([]) 
plt.yticks([])
img_path = './output/wordscloud.jpg'
img_dir = os.path.dirname(img_path)
os.makedirs(img_dir, exist_ok=True) 
plt.savefig(img_path, bbox_inches='tight', pad_inches=0,dpi=300)
plt.show()