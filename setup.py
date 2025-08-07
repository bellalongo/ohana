import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="ohana",
    version="0.0.1",
    author="Bella Longo",
    author_email="bellalongo.mail@gmail.com",
    description="A deep learning-based tool for detecting and segmenting cosmic rays in astronomical images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bellalongo/ohana",  # Link to your repository
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True, # This will include non-python files specified in MANIFEST.in
)