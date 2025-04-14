import jieba
import os
import chardet  # 新增编码检测库

# 加载自定义人名列表
with open('person_names.txt', 'r', encoding='utf-8') as f:
    person_names = [line.strip() for line in f]
for name in person_names:
    jieba.add_word(name)

input_dir = './data'
output_file = './corpus.txt'

with open(output_file, 'w', encoding='utf-8') as out_f:
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)

            # 动态检测文件编码
            with open(file_path, 'rb') as in_f_raw:
                raw_data = in_f_raw.read()
                encoding = chardet.detect(raw_data)['encoding']
                # 处理可能的检测失败（默认回退到 gbk）
                if not encoding:
                    encoding = 'gbk'

            with open(file_path, 'r', encoding=encoding, errors='ignore') as in_f:
                text = in_f.read().replace('\n', ' ')
                words = jieba.lcut(text)
                out_f.write(' '.join(words) + '\n')

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sentences = LineSentence('./corpus.txt')
model = Word2Vec(
    sentences,
    vector_size=100,  # 嵌入维度
    window=5,         # 上下文窗口大小
    min_count=3,      # 过滤低频词
    workers=4         # 并行线程数
)
model.save('jin_yong_word2vec.model')

# 加载模型和名称列表
model = Word2Vec.load('jin_yong_word2vec.model')
with open('person_names.txt', 'r', encoding='utf-8') as f:
    names = [line.strip() for line in f]

# 提取有效名称及其向量
valid_names = [name for name in names if name in model.wv]
vectors = [model.wv[name] for name in valid_names]

print(f"有效人物数量: {len(valid_names)}")

import numpy as np
from sklearn.decomposition import PCA

X = np.array(vectors)
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)  # 标准化

# 降维到2D和3D
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_normalized)

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_normalized)

import matplotlib
matplotlib.use('TkAgg')  # 添加在文件最开头
import jieba
import jieba.analyse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Songti SC', 'SimHei', 'STKaiti']  # 多个备选
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(15, 15))
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], alpha=0.5)
for i, name in enumerate(valid_names):
    plt.annotate(name, (X_pca2[i, 0], X_pca2[i, 1]), fontsize=8)
plt.title('金庸小说人物名称嵌入 - PCA 二维可视化')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.savefig('pca_2d.png', dpi=300)
plt.show()

import plotly.express as px

fig = px.scatter_3d(
    x=X_pca3[:, 0], y=X_pca3[:, 1], z=X_pca3[:, 2],
    text=valid_names,
    title='金庸小说人物名称嵌入 - PCA 三维可视化'
)
fig.update_traces(textposition='top center', marker_size=3)
fig.write_html("pca_3d.html")
fig.show()