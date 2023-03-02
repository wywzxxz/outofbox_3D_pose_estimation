[废弃]

此工程可以一键生成MHFormer结果

# 安装

- git submodule update
- 将MHFormer预训练权重放入models\MHFormer\checkpoint\pretrained，具体参考文件夹下readme
- 将预训练权重yolov3.weights、pose_hrnet_w48_384x288.pth放入models\MHFormer\demo\lib\checkpoint，具体参考文件夹下readme
- models\MHFormer\demo Line72：`ax.set_aspect('equal')`改为`ax.set_aspect('auto')`


# 使用
- videos文件夹下存放相应视频，格式参考sample
- cd src
- python MHFormer.py XXX
	- 其中XXX是videos文件夹下视频文件夹名，默认为sample
- videos/XXX文件夹下会生成结果以及相应2D、3D骨骼位置信息