from setuptools import setup, find_packages


install_requires=[]

setup(name='totalhuman',
        version='0.0.1',
        description='totalhuman gpu package',
        author='lululalamm',
        author_email='yms2109@gmail.com',
        python_requires='>=3',
        install_requires=install_requires,
        packages=find_packages(),
        include_package_data=True,
        )
