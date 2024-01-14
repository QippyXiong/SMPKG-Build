# SMPKG (Subway Maintenance Personnel Knowlege Graph) Build

## 关于 Build

使用文件 `setup.py`和 `Makefile`来build，把这两个文件放到你的工程目录下，建议python版本：3.11，3.8和3.12都不可build。

### Build所需应用程序

**On Linux：**

None （GCC，Python，已经装好了）

**On Windows**：(点击链接访问w)

- [Visual Studio Build Tools](https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools) （打开installer后点击生成工具的修改，勾选使用C++的桌面开发进行下载（6.6GB左右）
- [Git](https://git-scm.com/downloads)
- [Make](https://gnuwin32.sourceforge.net/downlinks/make.php)

需要添加Git和Make安装目录下的bin目录添加到Path环境变量。

Windows 执行make命令之前，需要执行 `sh`指令进入bash界面，这样就兼容了Makefile的执行。

```cmd.exe
X:xxx\xxx> sh
```

### Build文件修改

直接进到Makefile看好了，标注 `!修改!`的行请按照描述修改。

setup.py不用修改。

### Build 执行

需要下载 `Cython`，`mypy`两个包：

```bash
pip install cython mypy
```

如果是conda：

```bash
conda install cython mypy -c conda-forge
```

使用指令`make build PYTHON=xxx`build输出文件，如果python是当前shell默认python不用添加此参数。

在Windows下输出文件是每个python文件对应一个`.pyd`和`.pyi`，在Linux下是每个python文件对应一个`.so`和`.pyi`，前一个是代码编译的运行库，后一个是运行库接口文件。

在windows下使用git的 `sh`不能激活python环境，如果使用了虚拟环境而不是默认的python，需要指定环境python的路径，一般就在环境所在文件夹的底下。git `sh`里面磁盘路径不是 `C:`而是 `/c/`，比如说我要指定 `D:\program\miniconda3\envs\smpkg\python`，那么 `sh`里的路径是 `/d/program/miniconda3/envs/smpkg/python`，例子：

```bash
make build PYTHON='/d/program/miniconda3/envs/smpkg/python'
```

如果有入口文件的话，调用以下指令来运行build后的项目，Makefile中的ARGS用来输入参数：

```
make run PYTHON=xxx ARGS='xxx=xxx yyy=yyy'
```

#### FAQ

报错 9009 是 Python 未找到，说明`sh`里没找到名为python的环境变量，指定python路径即可。

报错 1 是 build 的python代码出了问题，如果是用pip下的mypy，cython下的可以尝试改用conda重下。

报错 2 是指定的python或者文件没找到，看看路径是否错了。

说找不到执行 build 之类的，要么是Makefile不在执行make的目录下，要么是下载的时候电脑自动在Makefile后面加了`.txt`。

## 本项目内容

下载本项目依赖之前，打开requirements.txt，修改pytorch-cuda对应的cuda版本为本机对应的cuda版本，cpu运行可以不下。

然后执行：

```shell
conda install --file requirements.txt -c pytorch -c nvidia -c conda-forge -c huggingface -c anaconda --yes
```

conda下载会比较缓慢，如果不能接受可以使用pip下载，需要先安装pytorch，然后再下载requiements.txt中的包。

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

接下来就可以运行了：

```bash
python main.py
```
