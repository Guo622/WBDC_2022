## 代码说明

### 环境配置

Python 版本：3.8
PyTorch 版本：1.10.0
CUDA 版本：11.3

所需环境在 `requirements.txt` 中定义。

### 数据

* 仅使用大赛提供的数据（有标注和无标注）。
* 未使用任何额外数据。

### 预训练模型

* 使用了 huggingface 上提供的 `hfl/chinese-roberta-wwm-ext-large` 模型。链接为： https://huggingface.co/hfl/chinese-roberta-wwm-ext-large

### 算法描述
` ` `
      融合了两个模型
` ` ` 
#### 模型一
* 对于视觉特征，使用`双向LSTM网络`得到视频时序特征 
* 对于文本特征，使用 `chinese-roberta-wwm-ext-large` 模型来提取特征。将title、asr、ocr三者进行等长(最大128)截断后拼接
  即 `[CLS] title [PAD].. [SEP] asr [PAD].. [SEP] ocr [PAD].. [SEP] `
  对 bert 输出做`mean pool`得到文本特征
* 用文本特征对视频时序特征做`CrossAttention`得到视频特征
* 将视屏特征和文本特征进行拼接经过一个`ConcatDenseSE`后，再用 MLP 结构去预测二级分类的 id

#### 模型二
* 文本和视频帧共享一个 bert，`chinese-roberta-wwm-ext-large `
* 将文本过一遍 bert 的 embedding 层，将视频帧过一个`线性映射层`再过 bert 的 embedding 层
* 过完 embedding 层后直接将特征在长度维度上拼接，再过 bert 的 encoder 层
* 对 encoder 输出做 `mean pool`，再用 MLP 结构去预测二级分类的 id

#### 融合
* 采用对score加权的方法，权重各为0.5，即 ![](http://latex.codecogs.com/svg.latex?score=score_1*0.5+score_2*0.5) 


### 性能

离线测试性能：模型一0.680左右、模型二0.689左右（均为单折）
B榜测试性能：融合模型 0.690783 （全量）


### 训练流程

* 模型一在无标注数据上进行 mlm 预训练
* 模型二在无边注数据上进行 mlm、mfm、itm 预训练
* 预训练完后再各自在有标注数据上进行有监督训练
* 模型一和模型二互不影响可分开训练，提供的train.sh为顺序执行，如果可并行训练，则执行
  ` pretrain1.py -> train1.py `
  ` pretrain2.py -> train2.py `


### 测试流程

* 线下划分10%的数据作为验证集。取验证集上最好的模型来做测试。
* 最后提交采用全量数据进行训练

### 部分代码参考
* https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st/tree/main/job6
* https://github.com/rsanshierli/Bert-Classification-EMA-AD
* https://github.com/z814081807/DeepNER/tree/master/src/utils
* https://github.com/WeChat-Big-Data-Challenge-2022/challenge
