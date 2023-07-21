# juejin_acrostic RNN通过训练古诗实现写藏头诗
配套文章讲解《谈谈RNN生成文本的小原理，动手实现AI写藏头诗》
https://juejin.cn/post/7257922419319291960

# 训练

训练必备文件：
```
main.py
poetry.txt
```
训练后会生成相关权重和数据。

# 仅生成

如果自己不训练，想直接使用我的权重，也提供下载地址

链接: https://pan.baidu.com/s/13cyAFooOdwYqmKT3wsygXQ?pwd=nxw4 提取码: nxw4 

下载后将tf文件夹、vocab.npy文件放到main.py同级目录下。然后main.py中main方法屏蔽调训练，直接生成即可。

```
if __name__ == "__main__":
  # 训练
  # train()
  # 预测演示 
  print(predict("掘金社区"))
```

生成必备文件：
```
main.py
vocab.npy
tf
-- checkpoint
-- checkpoint.data-00000-of-00001
-- checkpoint.index
```

