## 代码说明

### 数据

* 仅使用大赛提供的数据（有标注和无标注）。
* 未使用任何额外数据。

### 预训练模型

* 使用了 huggingface 上提供的 `hfl/chinese-macbert-base` 模型。链接为： https://huggingface.co/hfl/chinese-macbert-base

* 使用了 SwinTransformer 官方提供的 swin-tiny 模型。链接为：https://github.com/microsoft/Swin-Transformer

### 算法描述

* 对于视觉特征，使用 swin-tiny 提取视觉特征，最大帧数为12

* 将 title、asr、ocr三 者进行等长（最大84）截断后拼接

  即 `[CLS] title [PAD].. [SEP] asr [PAD].. [SEP] ocr [PAD].. [SEP] `

* 将拼接后的文本过一遍 bert 的 embedding 层，将视觉特征过一个线性映射层后再过 bert 的 embedding 层

* 将过完 embedding 层的文本特征和视觉特征在长度维度上拼接，再过 bert 的 encoder 层

* 对 encoder 输出做 mean pool ，再用 MLP 结构去预测二级分类的 id


### 训练流程

* 对 swin-tiny 进行 mfm 预训练
* 对 bert 进行 mlm 预训练
* 对整体模型进行 mlm、mfm、itm 预训练
* 对整体模型进行5折交叉验证训练

### 测试流程

* 取每一折的后两个 epoch 的模型进行 swa 获得融合模型进行推理

### 部分代码参考

* https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st/tree/main/job6
* https://github.com/rsanshierli/Bert-Classification-EMA-AD
* https://github.com/z814081807/DeepNER/tree/master/src/utils
* https://github.com/WeChat-Big-Data-Challenge-2022/challenge
* https://github.com/WeChat-Big-Data-Challenge-2022/challenge/tree/semi_submission