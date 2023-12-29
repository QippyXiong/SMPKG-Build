# 源代码所在目录，如果是当前目录写 . !修改!
PROJ_NAME = smpkg
# 编译输出目录		（不建议改）
BUILD_DIR = build
# 编译库结果输出目录 （不建议改）
OUTPUT_DIR = ${BUILD_DIR}/${PROJ_NAME}
# 编译lib时的临时目录（不建议改）
TEMP_DIR = ${BUILD_DIR}/temp
# 源代码文件列表 !修改!
SRC_FILES = ${PROJ_NAME}/*.py
# 源代码目录中不需要编译的文件 没有空着 !修改!
EXCULDE_FILES = ${PROJ_NAME}/__init__.py ${PROJ_NAME}/server.py ${PROJ_NAME}/logger.py
# 编译结果文件 (别改)
TARGET_BUILD_FILES =  ${BUILD_DIR}/$(basename $(SRC_FILES)).*.so
# 需要复制的文件，没有注释  !修改!
COPY_FILES = ${PROJ_NAME}/server.py ${PROJ_NAME}/logger.py
# 需要复制的文件夹，没有注释 !修改!
COPY_DIRS = database
# 代码程序入口	!修改!
ENTRY_POINT = main.py
# 复制文件目标目录
COPY_FILE_TARGET_DIR = ${BUILD_DIR}/${PROJ_NAME}
# 复制文件夹目录
COPY_DIR_TARGET_DIR = ${BUILD_DIR}
# 入口文件复制目录
COPY_ENTRY_POINT_TARGET_DIR = ${BUILD_DIR}
# 程序运行参数，没有空着 !修改!
ARGS ?=
# 如果有需求通过在make指令最后添加 PYTHON=/path_for_python/python.exe 来指定python路径
PYTHON ?= python

# 以下内容大概率不需要修改

export PY_CYTHON_BUILD_SRC_FILES=${SRC_FILES}
export PY_CYTHON_BUILD_EXCLUDE_FILES=${EXCULDE_FILES}

COPY_FILE = cp
COPY_DIR = cp -r


ifeq ($(OS),Windows_NT)
	TARGET_BUILD_FILES = ${BUILD_DIR}/$(basename $(SRC_FILES)).*.pyd
endif

print:
	@echo 'using shell: ' $(SHELL)
	@echo 'src files:' ${SRC_FILES}
	@echo 'target build output dir:' ${OUTPUT_DIR}
	@echo 'target build files:' ${BUILD_FILES}

run:
	cd ${BUILD_DIR} && $(PYTHON) ${ENTRY_POINT} ${ARGS}

cpfiles:
	@echo "copying files"
ifdef COPY_FILES
	$(COPY_FILE) ${COPY_FILES} ${COPY_FILE_TARGET_DIR}
endif
ifdef ENTRY_POINT
	$(COPY_FILE) ${ENTRY_POINT} ${BUILD_DIR}
endif
ifdef COPY_DIRS
	$(COPY_DIR) ${COPY_DIRS} ${COPY_DIR_TARGET_DIR}
endif


build_raw:
	$(PYTHON) setup.py build_ext -b ${BUILD_DIR} -t ${TEMP_DIR}

clean:
	@echo "cleaning up files"
	@rm -f ${PROJ_NAME}/**/*.c
	@rm -f ${PROJ_NAME}/*.c
	@rm -rf ${TEMP_DIR}

build: build_raw cpfiles clean
	@echo run 'make run' to run the program

test:
	$(PYTHON) ${ENTRY_POINT} ${ARGS}

default: build run