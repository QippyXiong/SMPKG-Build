from distutils.core import setup
from Cython.Build import cythonize

import os

proj_dir = 'smpkg'
exclude_files = [ 'server.py', 'logger.py' ]

src_files = [ proj_dir + os.path.sep + file for file in os.listdir(proj_dir) 
                if file.endswith('.py') and file not in exclude_files ]

setup(
    name="prompt",
    ext_modules=cythonize(
        src_files,
        language_level="3"
    )
)