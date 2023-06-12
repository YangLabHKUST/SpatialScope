from setuptools import setup,find_packages

setup(
    name='SpatialScope',               # 应用名
    version='0.0.1',              # 版本号
    description='A unified approach for integrating spatial and single-cell transcriptomics data by leveraging deep generative models',
    url='https://github.com/YangLabHKUST/SpatialScope',
    author='Xiaomeng Wan (xwanaf@connect.ust.hk), Jiashun Xiao (jxiaoae@connect.ust.hk)',
    license='GPLv3',
    packages=find_packages(),
    scripts=['src/Cell_Type_Identification.py','src/Nuclei_Segmentation.py', 'src/Singlet_Doublet_Classification.py', 'src/Decomposition.py','src/Train_scRef.py'],
    include_package_data=True,    # 启用清单文件MANIFEST.in
    exclude_package_date={'':['.gitignore']},
    install_requires = ['scanpy',
                       'squidpy',
                        'stardist',
                        'tensorflow',
                        'qpsolvers',
                        'torch',
                        'scikit-learn',
                       'pandas',
                       'matplotlib',
                       'scipy',
                       'ray']
)

