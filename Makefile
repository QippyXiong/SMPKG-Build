PROJ_NAME = smpkg
BUILD_DIR = build
OUTPUT_DIR = ${BUILD_DIR}/${PROJ_NAME}
TEMP_DIR = ${BUILD_DIR}/temp
SRC_FILES = ${PROJ_NAME}/*.py
BUILD_FILES =  ${BUILD_DIR}/$(basename $(SRC_FILES)).*.so # in linux is .so files
COPY_FILES = ${PROJ_NAME}/server.py ${PROJ_NAME}/logger.py
CONFIG_DIR = configs
COPY_FILE = cp
COPY_DIR = cp -r
PYTHON ?= python # use arg PYTHON to select interpreter

ifeq ($(OS),Windows_NT) # change shell and files for windows
	SHELL = cmd.exe
	SRC_FILES = ${PROJ_NAME}\*.py
	BUILD_FILES = .\${BUILD_DIR}\$(basename $(SRC_FILES)).*.pyd # in windows is .pyd files
	OUTPUT_DIR = .\${BUILD_DIR}\${PROJ_NAME}
	TEMP_DIR = .\${BUILD_DIR}\temp
	COPY_FILES = .\${PROJ_NAME}\server.py+.\${PROJ_NAME}\logger.py
	COPY_FILE = copy
	COPY_DIR = xcopy
endif

print:
	@echo $(SHELL)
	@echo 'src files:' ${SRC_FILES}
	@echo 'build files:' ${BUILD_FILES}

run:
	cd ${BUILD_DIR} && $(PYTHON) main.py

cpfiles:  # $(COPY_FILES)
	@echo "copying files"
	$(COPY_FILE) ${COPY_FILES} ${OUTPUT_DIR}
	$(COPY_FILE) main.py ${BUILD_DIR}
ifeq ($(OS),Windows_NT)
	$(COPY_DIR) database ${BUILD_DIR}\database /S /Y /Q
	$(COPY_DIR) ${CONFIG_DIR} ${BUILD_DIR}\${CONFIG_DIR} /S /Y /Q
else
	$(COPY_DIR) database ${BUILD_DIR}
	$(COPY_DIR) ${CONFIG_DIR} ${BUILD_DIR}
endif

build_raw:
	$(PYTHON) setup.py build_ext -b ${BUILD_DIR} -t ${TEMP_DIR}

clean :
ifeq ($(OS),Windows_NT)
	del .\${PROJ_NAME}\*.c
# del .\${PROJ_NAME}\*\*.c # May be you should do it by typing command
	rd ${TEMP_DIR} /S /Q
else
	@echo "cleaning up files"
	@rm -f ${PROJ_NAME}/**/*.c
	@rm -f ${PROJ_NAME}/*.c
	@rm -rf ${TEMP_DIR}
endif

build: build_raw cpfiles clean
	@echo run 'make run' to run the program

test:
	$(PYTHON) main.py

default: build run