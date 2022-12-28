运行`python main.py`即可将best_model.pt存的DNN转成MDNN并分别测试DNN和MDNN的精度，但是这种模式下并不考虑非理想特性。

如果要考虑非理想特性，可以添加参数`--nonideal <nonideal type>`,支持的输入包括`device|endurance|retention|finite|nonlinear`

如果要重新训练DNN，可以添加参数`--train`