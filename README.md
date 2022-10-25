////////////////////////////////////////////////////////////////
//Content
////////////////////////////////////////////////////////////////
DNN Models: 2xCNN 2xMLP
Datasets: 140 & 2071 Samples
Modules: Conv, Dense, Maxpooling, Softmax, ReLU (in nn_c dir)
Parallelization: CUDA & OpenMP


////////////////////////////////////////////////////////////////
//Instructions
////////////////////////////////////////////////////////////////
OpenMP:
cd lide_c_dnnlayers/test_c/ + model to be used
bash make.bash
bash run1.bash or bash run100.bash

CUDA:
cd lide_c_dnnlayers/test_cuda/ + model to be used
bash make.bash
bash run1.bash or bash run100.bash

Note: model directory with "more" suffix indicates dataset #2


////////////////////////////////////////////////////////////////
//Performance Data
////////////////////////////////////////////////////////////////
In Performance_Data directory
2x HPCToolkit database (serial version)
8x CUDA execution time outputs
8x OpenMP execution time outputs
