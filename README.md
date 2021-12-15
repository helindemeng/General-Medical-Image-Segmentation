# 通用的医学图像分割框架

二维医学图像分割

分割模型使用的是：U2Net



## 参数配置

配置文件： `config.py`



## 训练

模型训练：`u2net_train.py`



## 测试

模型测试：`u2net_test.py`



## 其他

查看损失：

- 使用记事本打开文件`visual_loss.bat`，将 `tensorboard --logdir logs` 中的logs修改为需要查看的目录，并保存。

  例如：`tensorboard --logdir logs-2021-07-22-10-19-56`

- 双击运行`visual_loss.bat`

- 运行后，在浏览器中输入网址  http://localhost:6006/ 可实时查看损失曲线和Dice系数变化曲线

