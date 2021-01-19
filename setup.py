import os
from setuptools import setup, find_packages


# Declare your non-python data files:
# Files underneath configuration/ will be copied into the build preserving the
# subdirectory structure if they exist.
data_files = []
for root, dirs, files in os.walk('configuration'):
    data_files.append((os.path.relpath(root, 'configuration'),
                       [os.path.join(root, f) for f in files]))

setup(
    name="proto",
    version="1.0",

    # declare your packages
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},

    # include data files
    data_files=data_files,

    root_script_source_version="default-only"
)
