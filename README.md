# 多机器人语义建图工作
该工作主要是基于NIPS2020的文章
[Object Goal Navigation using Goal-Oriented Semantic Exploration](https://arxiv.org/pdf/2007.00643.pdf)<br />
进行的改进工作，采用了他的Mapping模块，并将其扩展至多机器人领域以及iGibson仿真环境中。

## Installing Dependencies
- 安装多机版本的iGibson:[iGibson-MR](https://github.com/vsislab/iGibson-MR)
- 安装pytorch
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 
```

- 安装 [detectron2](https://github.com/facebookresearch/detectron2/) 用于语义分割模块:
```
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html 
```
- 安装项目需要的其余依赖（此处用pip可能会出问题，建议用conda）
```
conda install -r requirements.txt
```

## 一些接口的说明
目前已经完成了Mapping模块的搭建。
关于环境部分主要集中在`/env`目录下。
其中`multi_robot_mapping.py`即为对于iGibson核心类`iGibsonEnv`的继承，此处无需更多修改。
目录`/semantic_utils`下存放的两个函数，分别是语义分割模块和建图模块。

### Task函数
核心代码在`igibson_utils/semantic_mapping_task.py`，该模块是task函数，与任务有关的变量都存储于task类下。
最终得到的语义地图就是`self.full_map`和`self.local_map`两者都是以机器人起始位置为中心的格点地图（后者是前者的一部分），大小为机器人数目* 20 * 地图宽度 * 地图高度。
最终的local policy的模块也会集成在该类中。

## 运行与测试
根目录下的`test.py`。
目前简单写了一个test函数用于测试和调试，在该函数中可以可视化机器人的第一视角以及传回的地图。
运行参数见`arguments.py`中，可能会有多余的参数。

