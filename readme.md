# Get Started

## Installation

### Prerequisites (environment manager like conda, pipenv or poetry is recommended)

- python >= 3.9, < 3.13

### Using `pip` to install the suite from Pypi

```
pip install -r requirements.txt
```

# Run

## Access to data

First, please put datasets(AIOPS, UCR, TODS, NAB, Yahoo, WSD) under datasets folder

## Train and Test


### Handle a single dataset
```
./run.sh -d TODS
```

### Handle datasets
```
./run.sh -d "UCR,Yahoo"
```

### Baselines
Baselines can be found at https://github.com/dawnvince/EasyTSAD
Results can be found at http://adeval.cstcloud.cn
