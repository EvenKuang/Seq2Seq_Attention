# 研究生实验2：基于序列模型的英文到中文翻译机

标签： 2025 PostGraduate

---

You can click [here](https://www.zybuluo.com/photon058/note/2393314) to get the English version.

## 实验目的

1. 探索自然语言处理
2. 理解经典的Sequence-to-Sequence机器翻译模型
3. 掌握Attention机制在机器翻译模型上的应用
4. 搭建机器翻译模型，在简单小规模数据集上验证模型性能，培养工程能力
5. 了解Transformer在机器翻译任务上的应用

## 数据集
1. 本教程采用[中英文翻译数据集](http://www.manythings.org/anki/cmn-eng.zip)，更多翻译数据集可在[该网站](https://www.manythings.org/anki/)上下载
2. 共有23,610个翻译数据对，每对翻译数据在同一行：左边是英文，中间是中文，右边是其他属性信息，分割符是`\t`

## 实验环境
- [pytorch](https://pytorch.org/)
- [python3](https://www.python.org/)，至少包含下列python包：[sklearn](http://scikit-learn.org/stable/)，[numpy](http://www.numpy.org/)，[jupyter](http://jupyter.org/)，[matplotlib](https://matplotlib.org/)。
建议直接安装[anaconda3](https://anaconda.org/)，其已经内置了以上python包。
- [jieba](https://pypi.org/project/jieba/) (安装命令：`pip install jieba`)

## 实验步骤

基于注意力机制的机器翻译模型的示例代码可参考[Pytorch 教程](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html), 详细步骤如下：

本实验参考代码：[github](https://github.com/wujiaju/ML-07)

1. 下载[中英文翻译数据集](http://www.manythings.org/anki/cmn-eng.zip)，并解压为`./data/eng-cmn.tx`

2. 按行读取数据集，构建训练数据对时注意**移除**属性信息（每行只取前两个数据），否则会报错
  
3. 从训练句子中拆分出单词，构建数据集的中英文单词对照表
    注意: 默认`reverse=False`构建“英文-->中文”翻译器；感兴趣的同学也可尝试构建“中文-->英文”翻译器

4. 构建机器翻译模型：
    - 构建编码器（Encoder）
    - 构建基于注意力机制的解码器（Attention Decoder）

5. 定义损失函数，训练机器翻译模型

6. 使用`BLEU`等机器翻译指标评估已训练好的模型，可使用 [nltk 库](https://cloud.tencent.com/developer/article/1042161)
    ```
    # pip install nltk
    from nltk.translate.bleu_score import sentence_bleu
    bleu_score = sentence_bleu([reference1, reference2, reference3], hypothesis1)
    ```
7. 可视化测试结果,整理实验结果并完成实验报告.

[可选1] 感兴趣的同学可自行调整参数，如调整句子最大长度MAX_LENGTH，总训练次数n_iters，特征维度hidden_size等

[可选2] 感兴趣的同学可自行划分训练集/测试集,推荐的划分比例是 7:3，根据定性及定量的实验结果进一步分析模型性能

[可选3] 感兴趣的同学可自行探索使用[Transformer](https://arxiv.org/abs/1706.03762)完成任务，示例代码可参考[The Annotated Transformer blog](http://nlp.seas.harvard.edu/2018/04/03/attention.html)和[github](https://github.com/foamliu/Transformer)仓库 (注意：同样需要自行处理[中英文翻译数据集](http://www.manythings.org/anki/cmn-eng.zip))

整理实验结果并完成实验报告（实验报告模板将包含在[示例仓库](https://github.com/wujiaju/ML-01)中）。

---


# 实验代码及报告提交方式


## 提交内容

- 提交实验报告，不要求提交实验代码
- 实验报告需要按照模板编写，并导出成pdf文件（模板未必与本次实验内容完全契合，可适当修改模板章节）




