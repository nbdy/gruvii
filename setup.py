from setuptools import setup, find_packages

setup(
    long_description_content_type="text/markdown",
    long_description=open("readme.md", "r").read(),
    name="gruvii",
    version="0.42",
    description="music generation with tensorflow v2",
    author="Pascal Eberlein",
    author_email="pascal@eberlein.io",
    url="https://github.com/smthnspcl/gruvii",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords="music generation tensorflow gruv",
    packages=find_packages(),
)
