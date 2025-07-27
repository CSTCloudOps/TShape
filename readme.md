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

Datasets available from  https://anonymous.4open.science/r/TFC-Datasets-2703/

First, please put datasets(AIOPS, UCR, TODS, NAB, Yahoo, WSD) under datasets folder

## Train and Test

### Handle all default datasets
```
./run.sh
```

### Handle the specified data set
```
./run.sh NAB AIOPS
```

### Handle a single dataset
```
./run.sh Yahoo
```
