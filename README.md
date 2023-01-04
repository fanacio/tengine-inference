
# 关于tengine推理yolox模型整流程说明

## 1. 编译tengine（x86服务器编译和交叉编译两种）
```
编译tengine的整体流程如下：
安装配置cmake、G++、GCC等编译器--------->源码编译ACL库--------->源码编译protoc(libprotobuf-lite.so为编译模型生成工具所需的依赖)--------->源码编译Opencv库--------->配置ACL编译好的库到Tengine的Cmakelist中，打开ACL编译配置开关--------->源码编译Tengine

这里其他的编译过程不予赘述，参考本人OneNote笔记即可，且本人将编译的工具都放在了fanyichao_tengine:latest容器中了，如果迁移则直接迁移docker即可。
这里主要说明一下acl和tengine的编译过程
注意：本项目中的tengine是在服务器x86平台进行本地编译及交叉编译，在x86平台先生成tmfile离线模型（模型生成工具是本地编译出的，使用g++工具链），然后在arm架构开发板推理所用（推理库和头是交叉编译出的，使用g++-aarch64-linux-gnu交叉编译工具链）。
注意：查看编译出来的库是否为arm架构或者x86架构可以执行命令： file libprotobuf.so.18.0.1
输出结果：libprotobuf.so.18.0.1: ELF 64-bit LSB shared object, x86-64, version 1 (GNU/Linux), dynamically linked, BuildID[sha1]=57acd4dc81cc9d3f5786a9124e6e3f582780909b, with debug_info, not stripped
```

### 1.1 编译acl
- 用途： acl即为arm COMPUTE LIBRARY，在这里作为tengine后端加速使用，所以需要提前编译出库和头。参考tengine官方提供的文档得知"Tengine支持与ACL的OpenCL库集成，通过ARM-Mail GPU对CNN进行推理。"，换言之，如果arm边缘设备上没有mail-gpu，则不能进行acl编译，也不能用acl进行推理。
- 官网链接： https://arm-software.github.io/ComputeLibrary/latest/operators_list.xhtml
- 编译：进入ComputeLibrary目录，执行complie.sh脚本即可。编译获得的库和头均保存在ComputeLibrary/install目录下，且分别为include、lib目录

### 1.2 编译tengine源码
- 用途：其一，编译生成tengine三方库和头文件用于推理模型（交叉编译）；其二，编译生成模型转换工具，用于将训练的模型转换成tmfile模型（本地编译）
- 官网链接：
    - cgithub网址：https://github.com/OAID/Tengine
    - 中文文档：https://tengine.readthedocs.io/zh_CN/latest/
- 编译：
```
首先需要明确如何编译（arm编译、x86编译、交叉编译等）、编译什么（三方库、模型生成工具、量化工具等）这些都需要通过最外面的 Tengine-lite/CMakeLists.txt设置开关状态来决定，我这里为了方便直接执行脚本即可。
且需要注意的是TENGINE_BUILD_CONVERT_TOOL和TENGINE_ENABLE_ACL不能同时编译，因为TENGINE_BUILD_CONVERT_TOOL支持TENGINE_OPENMP开关量为ON，而TENGINE_ENABLE_ACL不支持，且TENGINE_BUILD_CONVERT_TOOL为本地编译，TENGINE_ENABLE_ACL为交叉编译。为了避免后期错误，这里设置脚本形式编译：即写两个脚本compile.sh和compile_convert_tool.sh，分别编译生成三方库和模型转换工具。
```    
  - step1：在编译之前，需要先在Tengine-lite目录下创建一个文件夹，名为3rdparty/acl，并把1.1中编译生成的acl库和头放在此文件夹下；
  - step2：分别赋予compile_convert_tool.sh和compile.sh权限：chmod 777 -R ***；
  - step3：分别执行compile_convert_tool.sh和compile.sh脚本；
  - step4：查看并检查生成的库和工具。
  ```
  >>1.2.1.查看并检查模型转换工具
        进入Tengine-lite/build-convert_tool/install目录下有bin、include、lib三个文件夹，则bin中存放了模型生成工具，名为convert_tool，执行命令file convert_tool查看是否为x86-64版本。
  >>1.2.2.查看并检查三方库及头文件
        进入Tengine-lite/build-aclm/install目录下有include、lib两个文件夹，则lib中包含了生成的动态库和静态库，命名为libtengine-lite-static.a/so，执行命令file libtengine-lite.*查看是否为ARM aarch64版本。
  ```
  如果没有问题，则自此生成了转tmfile模型所需要的工具，以及arm板子上推理所需要的include和lib文件

## 2. 生成tmfile离线模型
- 说明：我下载的这版Tengine源码中（包含自己新增的算子），目前算子比较全面的模型转换工具为darknet工具和caffe工具，且yolox目前使用的模型转换工具链为pytorch->caffe->tmfile，故使用caffe工具接口。（针对yolov4本人也完善开发了darknet版本工具，能够正确生成tmfile模型）
- 使用方法：可以查看tengine中文文档中关于"模型转换工具"->"执行模型转换"部分，也可以直接执行./convert_tool --help查看帮助。
```
        [Convert Tools Info]: example arguments:
                ./convert_tool -f onnx -m ./mobilenet.onnx -o ./mobilenet.tmfile
                ./convert_tool -f caffe -p ./mobilenet.prototxt -m ./mobilenet.caffemodel -o ./mobilenet.tmfile
                ./convert_tool -f mxnet -p ./mobilenet.params -m ./mobilenet.json -o ./mobilenet.tmfile
                ./convert_tool -f darknet -p ./yolov3.weights -m ./yolov3.cfg -o yolov3.tmfile
```
- 调试：一般而言，为了验证模型生成过程中哪个算子存在问题，则需要调试。调试时将模型的类型、路径都写在了Tengine-lite/tools/convert_tool/convert_tool.cpp中。
- 模型存放位置：Tengine-lite/models
```
以生成phonedet模型为例，
执行命令：./convert_tool -f caffe -p ../../../models/phonedet/phonedet.prototxt -m ../../../models/phonedet/phonedet.caffemodel -o ../../../models/data/phonedet.tmfile
执行结果：Convert model success. ../../../models/phonedet/phonedet.caffemodel -----> ../../../models/data/phonedet.tmfile
表示生成模型
```
注意：这里也可以用量化工具将其量化，本人写了在acl中写了一些支持uint8量化后数据类型的算子，但目前而言依旧存在不兼容的接口需要开发，后续完善。

## 3. yolox推理工程编译
- 编译环境：不同于上述1和2小节，此小节是直接部署在推理平台的。因为目前实验的平台是arm架构的RK3399开发板（ubuntu18.04系统）,故直接在RK3399上编译。
- 推理工程示例名称：tengine_inference_phonedet（以手机检测为例）
- 操作流程：
```
step1：首先将1.1小节中生成的acl库和头文件复制到tengine_inference_phonedet/3rdparty/aclm目录下（因为在开发tengine的acl后端时为了不破坏源代码就重新创建了一个aclm后端分支）；然后将1.2小节中生成的tengine头和库复制到tengine_inference_phonedet/3rdparty/tengine目录下；
step2：将2小节中生成的离线模型复制到tengine_inference_phonedet/models目录下；
step3：按照tengine_inference_phonedet工程中的README.md文件进行操作即可。
注意：1.run_sample.sh脚本中写了调用库的临时路径，如果迁移工程可能导致链接找不到的问题，遇到此问题则在这里修改即可；2.此工程是中没有开发解码操作，为了方便，直接将解码保存为bin文件；3.执行前赋予脚本可执行权限。
```
```
推理性能如下所示：
Repeat 100 times, avg time 120.14 ms, max_time 222.62 ms, min_time 70.73 ms
 0:  90%, [ 373,  392,  570,  495], phone
 0:  95%, [ 764,  496, 1089,  656], phone
```

- 小技巧
#### 这里有一个提高推理性能的技巧，即提高mali-gpu频率：
```
    对于系统内部的文本文件，如果我们需要对其进行修改，我们不能在自己用户下进行修改，需要去su超级用户下进行修改：

    su
    输入密码
    执行： echo xxxxx > file

    其中，xxxxx表示写入的内容，file表示要写入的目标文件

    例如：
    RK3399中需要修改mali-GPU的最低运行频率，需要进入如下设置：
    进入su用户
    cd /sys/devices/platform/ff9a0000.gpu/devfreq/ff9a0000.gpu/
    echo 800000000 > min_freq
```

## 4. 模型转换说明
- 用途：获取caffe模型，即pytorch->caffe
- 使用说明：参考pytorch-phonedet工程中的README.md文件
- 对应工程示例名称：pytorch-phonedet（以手机检测为例）
- 执行环境：yolox容器，获取对应镜像的方法在下节
- 注意点：pytorch-phonedet中包含了pytorch2caffe工具，此工具命名为Caffe和Caffe_20221018两个版本的工具，此工具可以迁移到其他工程中使用，使用方法参考pytorch-phonedet/demo.py中用法
```
通过模型转换，能够获得phonedet.caffemodel文件和phonedet.prototxt文件，将其移动到Tengine-lite/models目录下，按照第2小节中的操作即可生成tmfile模型（先看下面注意点）。
```
- 注意点：
```
获得的phonedet.prototxt文件不能直接拿去执行Tengine-lite/build-convert_tool/install/bin/convert_tool生成tmfile，有些细节的地方需要收到修改，在Tengine-lite/tools/convert_tool/caffe/te_caffe.proto中定义了转模型时候phonedet.prototxt的写法，需要按照此标准对其进行细小修改，如下修改：
>>1.upsample层的scale改为整形（也可以对te_caffe.proto标准进行修改）
>>2.更改头部信息，将其看作一个layer进行解析。
```

## 5. 其他且重要

### 5.1 工具（镜像）
- 编译acl及tengine的镜像
```
获取工具的方式：
本地目录下：命名为fanyichao_tengine.tar
本人的dockerhub直接pull：命令为docker pull fanacio/fanyichao_tengine
```
- 调试pytorch训练工程的镜像
```
获取工具的方式：
本人的dockerhub直接pull：命令为docker pull fanacio/yolo_yolox
或者直接找一个包含python依赖库的docker即可。
```

### 5.2 关于RK3399固件升级及刷机
- 详情参见《RK3399刷机资料/RK3399固件升级.docx》

### 5.3 关于tengine + acl后端的推理
- 参见1.1用途，acl需要mail-gpu，所以在jetson上无法运行acl版本的tengine_inference_phonedet推理工程，只能执行cpu版本的推理工程，即#define ACLM  false
```
在jetson平台，可以编译acl源码（arm版），也可以编译生成tengine库（arm版），但是在运行推理工程的时候会报错如下信息：
```
```
    Can't load libOpenCL.so: libOpenCL.so: cannot open shared object file: No such file or directory
    Can't load libGLES_mali.so: libGLES_mali.so: cannot open shared object file: No such file or directory
    Can't load libmali.so: libmali.so: cannot open shared object file: No such file or directory
    Couldn't find any OpenCL library.
    terminate called after throwing an instance of 'std::runtime_error'
    what():  in create_opencl_context_and_device src/runtime/CL/CLHelpers.cpp:128: !opencl_is_available()
```
```
试图去全局搜索libOpenCL.so库，但没有找到，这是因为这个系统本身没有mail-gpu对应的库，即在RK3399中对应的/root-ro/etc/OpenCL/vendors/mali.icd下存在libMaliOpenCL.so，所以RK3399可以使用ACL后端。
另外，对于jetson边缘设备而言可以使用TensorRT后端。
```