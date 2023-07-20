#%%
# 导入必要的包
import tensorflow as tf
import numpy as np
import os

# 读取词汇库
vocab= []
if os.path.exists('vocab.npy'):
  vocab = np.load('vocab.npy')
# 词集的长度,也就是字典的大小
vocab_size = len(vocab)
# 将众多输入输出对打散，并64个为一组
BATCH_SIZE = 64
# 嵌入的维度，也就是生成的embedding的维数
embedding_dim = 256
# RNN 的单元数量
rnn_units = 1024
# 检查点保存至的目录
checkpoint_dir = "tf2/checkpoint"
# 读取文本并整理数据集
def get_dataset():
  with open('poetry.txt', 'r', encoding='utf-8') as file:
      text = file.read()
  # 列举文本中的非重复字符
  vocab = sorted(set(text))
  vocab_size = len(vocab)
  # 保存字典，等预测时直接加载使用，不用再次计算
  np.save('vocab.npy',vocab)
  # 创建从非重复字符到索引的映射
  char2idx = {u:i for i, u in enumerate(vocab)}
  # 创建从索引到非重复字符的映射
  idx2char = np.array(vocab)
  # 将训练文件内容转换为索引的数据
  text_as_int = np.array([char2idx[c] for c in text])
  # 创建训练样本，将转化为数字的诗句外面套一层壳子，原来是[x]
  char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
  # 所有样本中，每6个字作为一组，其中包含一个空格
  sequences = char_dataset.batch(6, drop_remainder=True) # 数据当前状态：((6,x))
  # 将每5个字作为一组所有样本，掐头去尾转为输入，输出结对
  dataset = sequences.map(split_input_target) # 数据当前状态：((4,x), (4,x))
  dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True) # 数据当前状态：((64, 4), (64, 4))
  return dataset

# 处理一句段文本，拆分为输入和输出两段
def split_input_target(chunk):
    chunk = chunk[:-1] # 去掉空格
    input_text = chunk[:-1] # 尾部去一个字
    target_text = chunk[1:] # 头部去一个字
    return input_text, target_text # 窗前明月光 变为：窗前明月&前明月光

# 构建模型
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)])
    return model
# 训练数据
def train():
  dataset = get_dataset()
  # 整一个模型
  model = build_model(
    vocab_size = len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)
  # 损失函数
  def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
  # 配置优化器和损失函数
  model.compile(optimizer='adam', loss=loss)
  # 保存回调
  checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True)
  # 进行训练
  history = model.fit(dataset, epochs=100, callbacks=[checkpoint_callback])

#%% 预测数据
def predict(start_string):
  # 创建从非重复字符到索引的映射
  char2idx = {u:i for i, u in enumerate(vocab)}
  # 创建从索引到非重复字符的映射
  idx2char = np.array(vocab)
  # 构建模型
  model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
  # 当初只保存了权重，现在只加载权重
  model.load_weights("tf2/checkpoint")
  # 从历史结果构建起一个model
  model.build(tf.TensorShape([1, None]))
  num_generate = 4   # 生成的字符数

  texts = []
  for c in start_string:
    input_eval = [char2idx[s] for s in c]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 0.7  # 控制生成的随机性
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions /= temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    texts.append(c + ''.join(text_generated))
  return ' '.join(texts)

#%%
if __name__ == "__main__":
  # 训练
  train()
  # 预测演示 
  """
  掘井须到头 金刀旧剪荒 社过花委树 区区趁适侣
  掘井似翻罗 金钗已十年 社日穿痕月 区区趁试期
  掘井须到头 金刀剪紫绒 社日放歌还 区区趁适来
  掘井须到南 金钿坠芳草 社会前年别 区区趁试期
  掘井须到头 金刀旧剪成 社酒收杯杓 区区趁适肠
  掘井须到三 金钏色已歇 社客加笾食 区区趁试期
  """
  print(predict("掘金社区"))
# %%
