from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'auto_drive'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # install picture assets so runtime can find them via get_package_share_directory
        (os.path.join('share', package_name, 'picture'),
         glob(os.path.join('picture', '*.png')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='75562992+Whathelp233@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [ 
            'auto_drive_node = auto_drive.auto_drive_node:main',
            'auto_drive_node_traffic = auto_drive.auto_drive_node_traffic:main'
        ],
    },
)
