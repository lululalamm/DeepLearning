from setuptools import setup, find_packages


install_requires=[]

setup(name='totalface_cpu',
        version='0.0.3',
        description='totalface cpu package',
        author='lululalamm',
        author_email='yms2109@gmail.com',
        python_requires='>=3',
        install_requires=install_requires,
        packages=find_packages(),
        include_package_data=True,
        )
