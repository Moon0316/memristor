### 项目环境

本项目包依赖`memtorch`库运行。在安装`memtorch`时可能需要`libeigen`库辅助编译。

### 忆阻器模拟相关代码

`plot_memristor_properties.py`：为捏滞回线可视化的实现，默认测试代码为`memtorch`库中的`VTEAM`模型。

`ideal_mem_resistor.py`：实现了理想忆阻器线性漂移子模型并测试了不同频率正弦激励下的忆阻器图线。

以上文件均可直接运行。

### Memtorch 离线训练

运行`python main.py`即可将`best_model.pt`存的DNN转成MDNN并分别测试DNN和MDNN的精度，但是这种模式下并不考虑非理想特性。

如果要考虑非理想特性，可以添加参数`--nonideal <nonideal type>`,支持的输入包括`device|endurance|retention|finite|nonlinear`

如果要重新训练DNN，可以添加参数`--train`, 训练参数还有`--lr`,`--step_size`,`--epoch`,`--batch_size`

设置参数`--dataset <dataset name>`可以选择用于训练/测试模型的数据集，支持的类型包括`minist|cifar10`

### 1T1R 在线训练

运行`python 1T1R.py --use_sign` 即可运行1T1R阵列模拟算法，器件相关参数以及超参数可以通过args修改