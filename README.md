# MLSE
[![Build Status](https://travis-ci.com/rsamer/MLSE.svg?token=FAdLEvwwnf8nbptujfqf&branch=master)](https://travis-ci.com/rsamer/MLSE)

Requirements:
- Python 2.7
- numpy
- scipy
- matplotlib
- nltk
- scikit-learn
- beautifulsoup4

Tested with:
- Python 2.7
- numpy 1.9.2
- scipy 0.15.1
- matplotlib 1.4.3
- nltk 3.0.3
- sklearn 0.16.1
- beautifulsoup 4.3.2

Setup for Ubuntu 16.04 LTS:
```sh
sudo apt-get install python2.7
sudo apt-get install python-minimal
sudo apt-get install python-pip
sudo apt-get install libfreetype6-dev libpng-dev
sudo apt-get install python-dev
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
bash miniconda.sh -b -p /home/TODO_INSERT_YOUR_USERNAME_HERE/miniconda
export PATH=/home/TODO_INSERT_YOUR_USERNAME_HERE/miniconda/bin:$PATH
conda update --yes conda
conda install --yes atlas numpy scipy matplotlib nose
pip install nltk
pip install -U scikit-learn
pip install beautifulsoup4
```

Usage:
```sh
cd src/
python -m main ../data/example
```
