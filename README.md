## MilkywayMap

本项目是基于Unity，用C#脚本通过读取真实恒星星表，实现银河地图模拟的探索项目，目前发现存在远处恒星抽搐闪动的问题，目前大概知道是因为浮点数精度的原因导致的，尚未解决。


## 2026-08-15更新

新上传的yinheditu_jindu1和yinheditu_jindu2两个文件夹是在conda环境（python=3.10 仅安装有numpy、matplot、pillow的情况下）中实现的程序化生成银河地图模拟探索。

jindu1是2d的，jindu2是3d的（jindu2运行main.py时不生成整个星系的3d，单独运行weitushengcheng_yinheditu3d.py时生成整个星系的3d）。

这个项目是学习pioneerspacesim[https://github.com/pioneerspacesim/pioneer]星图的实现而产生的。

# jindu2的weitushengcheng.py的思路如下：

根据bmp密度图生成星系的密度，根据密度决定恒星的多少；图是2d的，所以厚度方向，引入一个Z轴衰减系数实现星系的厚度；将整个星系空间分成一块一块的扇区，固定住随机种子，确保了只要坐标不变，生成的恒星位置、类型等属性就永远相同；基于这个种子，程序根据权重表为恒星随机分配其类型、温度、名字等属性；画成3d图时，通过LOD技术实现随视场缩放显示不同细节，当视场放大到一定倍数时，卸载远处的恒星不再渲染节省计算机资源，同时显示近处的恒星的真实颜色和名字；3d图默认可以围绕原点旋转放大。

# 下一步的开发计划：

1.加上视锥剔除

2.把Sol 500ly 以内换成从json读取的真实恒星，500ly以内不再程序生成

3.加上绿色的坐标参考网格，网格本身尺寸固定，但根据当前视图的缩放级别决定是否显示

# 运行方法：

1.下载这个文件夹解压到你喜欢的D盘（解压后应该是这样 D:/yinheditu_jindu2 里边直接是main.py这个层级）

2.下载安装miniconda，运行miniforge prompt

3.输入 conda create -n pioneerenv python=3.10 -y 回车

4.输入 conda activate pioneerenv 回车

5.导航到D盘

6.cd到yinheditu_jindu2文件夹里

7.输入 idle 回车

8.通过ilde再打开运行相应的.py文件



