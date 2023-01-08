### 项目环境

本项目包依赖`memtorch`库运行。在安装`memtorch`时可能需要`libeigen`库辅助编译。`memtorch`库安装参考 https://github.com/coreylammie/MemTorch/blob/master/README.md 

### 忆阻器模拟相关代码 

1. visualize文件中`plot_memristor_properties.py`：为捏滞回线可视化的实现，默认测试代码为`memtorch`库中的`VTEAM`模型。

2. visualize文件中`ideal_mem_resistor.py`：实现了理想忆阻器线性漂移子模型并测试了不同频率正弦激励下的忆阻器图线。


### 基于Memtorch 离线训练

models文件夹下保存着离线训练好的模型"mnist_best_model.pt","cifar10_best_model.pt"，由于文件大小限制，在本repo中我们并未将模型上传，可自行上传训练好的模型或与我们联系。提交的压缩文件中已包含。

1. 运行`python train_offline/main.py`即可将`best_model.pt`存的DNN转成MDNN并分别测试DNN和MDNN的精度，但是这种模式下并不考虑非理想特性。

2. 如果要考虑非理想特性，可以添加参数`--nonideal <nonideal type>`,支持的输入包括`device|endurance|retention|finite|nonlinear`

3. 如果要重新训练DNN，可以添加参数`--train`, 训练参数还有`--lr`,`--step_size`,`--epoch`,`--batch_size`

4. 设置参数`--dataset <dataset name>`可以选择用于训练/测试模型的数据集，支持的类型包括`mnist|cifar10`


### 忆阻阵列在线训练——基于双向电导连续调制

运行`python train_online2/main.py --use_sign` 即可运行1T1R阵列模拟算法，器件相关参数以及超参数可以通过args修改。--add noise加入可以使得参数更新时加上随机噪声，--sigma可用来控制加入噪声方差。

### 忆阻阵列在线训练——基于充放电机制
运行`python train_online1/single_layer.py`可运行模拟忆阻阵列在单层感知机上的训练过程
