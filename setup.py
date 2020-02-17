import setuptools


setuptools.setup(
        name='libertem-hackaton',
        version='0.0.1',
        author='',
        author_email='',
        description='Libertem integration into Nion Swift',
        packages=['nionswift_plugin/libertem_UI'],
        install_requires=['nionswift', 'liberTEM'],
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Programming Language :: Python :: 3.7',
        ],
        include_package_data=True,
        python_requires='~=3.7',
        )