# SMPKG (Subway Maintenance Personnel Knowlege Graph) Build

## 关于 Build

使用文件 `setup.py`和 `Makefile`来build，建议python版本：3.11，3.8和3.12都不可build。

### Build所需应用程序

**On Linux：**

    None （GCC，Python，已经装好了）

**On Windows**：(点击链接访问w)

- [Visual Studio Build Tools](https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools)
- [Git](https://git-scm.com/downloads)
- [Make](https://gnuwin32.sourceforge.net/downlinks/make.php)

需要添加Git和Make安装目录下的bin目录添加到Path环境变量。

Windows 执行make命令之前，需要执行 `sh`指令进入bash界面，这样就兼容了Makefile的执行。

```cmd.exe
X:xxx\xxx> sh
```

后续两个操作系统不会有区别。

### Build 修改文件

直接进到Makefile看好了，标注 `!修改!`的行请按照描述修改。

setup.py不用修改。

### Build 执行

build输出，如果python是当前shell默认python不用添加此参数。

```shell
make build PYTHON=xxx
```

如果有入口文件的话，调用以下指令，Makefile中的Args用来输入参数

```
make run PYTHON=xxx ARGS='xxx=xxx yyy=yyy'
```

## 本项目内容

下载本项目依赖之前，打开requirements.txt，修改pytorch-cuda对应的cuda版本为本机对应的cuda版本，cpu运行可以不下。

然后执行：

```shell
conda install --file requirements.txt -c pytorch -c nvidia -c conda-forge
```

接下来就可以运行了。
