from distutils.core import setup
from Cython.Build import cythonize

import os
import glob

raw_src_files = os.environ['PY_CYTHON_BUILD_SRC_FILES'].replace('\t', '').strip().split(' ')
if 'PY_CYTHON_BUILD_EXCLUDE_FILES' in os.environ:
    exclude_files = os.environ['PY_CYTHON_BUILD_EXCLUDE_FILES'].replace('\t', '').strip().split(' ')
else:
    exclude_files = []

build_src_files = []
for file in raw_src_files:
    if file.endswith('.py') and file not in exclude_files:
        list_files = [ file_path.replace('\\', '/') for file_path in glob.glob(file, recursive=True) ]
        build_src_files.extend(list_files)

build_exculde_files = []
for file in exclude_files:
    if file.endswith('.py'):
        list_files = [ file_path.replace('\\', '/') for file_path in glob.glob(file, recursive=True) ]
        build_exculde_files.extend(list_files)


# 生成源文件列表
src_files = [ 
    file for file in build_src_files 
    if file.endswith('.py') and file not in build_exculde_files 
]


def genpyi():
    r"""  generate pyi files """
    from mypy import stubgen
    # 3.8 版本及以下此代码会报错
    stubgen.main([ *src_files, '--export-less', '-o', 'build'])


setup(
    name="smpkg",
    ext_modules=cythonize(
        src_files,
        language_level="3"
    )
)

if __name__ == "__main__":
    genpyi()
