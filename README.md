# ugatit-paddle

# 本实例是用paddle库来实现的

    conda环境为python=3.7    paddlepaddle-gpu=1.8.0

# 如何运行该模型

第一步：下载selfie2anime数据集并解压文件

    unzip /selfie2anime.zip -d ./data/selfie2anime
  
第二步：生成数据集路径的txt文件

    python get_all_imgs_txt.py
  
第三步：开始训练模型

    python train.py
  
# 可能出现的问题

一：路径不正确或显示无该文件

    请修改代码里的文件路径，确保是你本地的文件路径
  
二：出现ImportError: cannot import name 'imsave' from 'scipy.misc'错误

    请运行一下命令
    pip uninstall scipy -y
    pip uninstall pillow -y
    pip install pillow==5.2.0
    pip install scipy==1.1.0
