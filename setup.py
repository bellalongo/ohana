import setuptools

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of your requirements file
with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="ohana",
    version="0.0.1",
    author="Bella Longo",
    author_email="bellalongo.mail@gmail.com",
    description="A deep learning-based tool for detecting and segmenting cosmic rays in astronomical images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bellalongo/ohana",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8', 
    include_package_data=True,

    entry_points={
        'console_scripts': [
            'ohana-create-training=ohana.scripts.create_training_set:main',
            'ohana-train=ohana.scripts.train_model:main',
            'ohana-predict=ohana.scripts.run_prediction:main',
        ],
    },
)