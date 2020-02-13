# Siamese Network (using tensorflow) on Quora duplication questions problem
Text Siamese Network provides a CNN based implementation of Siamese Network to 
solve Quora duplicate questions identification problem.
Quora question pair dataset has ~400k question pairs along with a binary label 
which states whether a pair of questions are similar or dissimilar. 
The Siamese Network based tries to capture the semantic similarity between questions.

> Siamese Network 孪生网络
> 数据和训练后的模型[下载地址](https://yunpan.360.cn/surl_yraIb8WaIA4)

## Requirements
- Python 2.7
- Pip
- tensorflow 1.8
- FastText
- Faiss

### Environment Setup
Execute requirements.txt to install dependency packages
```bash
pip install -r requirements.txt
```

> 现在 > tf 2.0，与之前的版本不兼容，可能会遇到各种问题。如下我搭建的环境。

### 环境搭建
```shell script
conda create -n tf18 python=2.7
source activate tf18
pip install tensorflow-cpu==1.8.0
```
下面是本人所有pip安装包。
```text
Package                            Version
---------------------------------- -----------
-atplotlib                         2.2.5
absl-py                            0.9.0
appnope                            0.1.0
astor                              0.8.1
attrs                              19.3.0
backports-abc                      0.5
backports.functools-lru-cache      1.6.1
backports.shutil-get-terminal-size 1.0.0
backports.weakref                  1.0.post1
bleach                             1.5.0
certifi                            2019.11.28
configparser                       4.0.2
contextlib2                        0.6.0.post1
cycler                             0.10.0
decorator                          4.4.1
defusedxml                         0.6.0
entrypoints                        0.3
enum34                             1.1.6
faiss-cpu                          1.6.1
fasttext                           0.9.1
funcsigs                           1.0.2
functools32                        3.2.3.post2
futures                            3.3.0
gast                               0.3.3
grpcio                             1.26.0
html5lib                           0.9999999
importlib-metadata                 1.5.0
ipaddress                          1.0.23
ipykernel                          4.10.1
ipython                            5.9.0
ipython-genutils                   0.2.0
ipywidgets                         7.5.1
Jinja2                             2.11.1
jsonschema                         3.2.0
jupyter                            1.0.0
jupyter-client                     5.3.4
jupyter-console                    5.2.0
jupyter-core                       4.6.1
kiwisolver                         1.1.0
Markdown                           3.1.1
MarkupSafe                         1.1.1
mistune                            0.8.4
mock                               3.0.5
nbconvert                          5.6.1
nbformat                           4.4.0
notebook                           5.7.8
numpy                              1.16.6
pandas                             0.24.2
pandocfilters                      1.4.2
pathlib2                           2.3.5
pexpect                            4.8.0
pickleshare                        0.7.5
pip                                19.2.3
prometheus-client                  0.7.1
prompt-toolkit                     1.0.18
protobuf                           3.11.3
ptyprocess                         0.6.0
pybind11                           2.4.3
Pygments                           2.5.2
pyparsing                          2.4.6
pyrsistent                         0.15.7
python-dateutil                    2.8.1
pytz                               2019.3
pyzmq                              18.1.1
qtconsole                          4.6.0
scandir                            1.10.0
scikit-learn                       0.20.4
scipy                              1.2.3
Send2Trash                         1.5.0
setuptools                         41.4.0
simplegeneric                      0.8.1
singledispatch                     3.4.0.3
six                                1.14.0
sklearn                            0.0
subprocess32                       3.5.4
tensorboard                        1.8.0
tensorflow                         1.8.0
termcolor                          1.1.0
terminado                          0.8.3
testpath                           0.4.4
tornado                            5.1.1
traitlets                          4.3.3
wcwidth                            0.1.8
Werkzeug                           1.0.0
wheel                              0.33.6
widgetsnbextension                 3.5.1
zipp                               1.1.0
```

## Training
1. Quora questions dataset is provided in ./data_repository directory. 
2. To train 
```bash
python train_siamese_network.py
```
> 生成如下文件：
> 生成ft_skipgram_ws5_dim64.bin 
> 生成metadata.tsv 
> 生成 model.*
 
## Prediction
Open Prediction.ipynb using Jupyter Notebook to look into Prediction module.

## Results
Given Question: **"Is it healthy to eat egg whites every day?"** most similar questions are as follows:
1. is it bad for health to eat eggs every day
2. is it healthy to eat once a day
3. is it unhealthy to eat bananas every day
4. is it healthy to eat bread every day
5. is it healthy to eat fish every day
6. what high protein foods are good for breakfast
7. how do you drink more water every day
8. what will happen if i drink a gallon of milk every day
9. is it healthy to eat one chicken every day
10. is it healthy to eat a whole avocado every day

Due to limitation in max file size in git, I haven't uploaded trained model in git. 
You can download pre-trained model from 
[here](https://drive.google.com/drive/folders/1FEdvcQt-tbNCZeUKhawFxyAn6Dn7H08I?usp=sharing) and 
unzip and paste pre-trained model to "./model_siamese_network" directory.

## Note
To train on a different dataset, you have to build a dataset consisting of similar and 
dissimilar text pairs. 
Empirically, you need to have at least ~200k number of pairs to achieve excellent performance. 
Try to maintain a balance between similar and dissimilar pairs [50% - 50%] is a good choice. 
