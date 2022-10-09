# pos-agreement

## Prerequisites

The following packages are used in this project:
- jsonlines
- nltk
- numpy

You can install them by running:
```shell
pip install jsonlines nltk numpy
```

## Quick Start

To calculate agreement scores, simply run the following commands:

```shell
git clone https://github.com/Liskie/pos-agreement.git
cd pos-agreement
python main.py
```

The default data directory is `data` under the project root.
You may change it via the `-d` or `--data_dir` parameter like:
```shell
python main.py -d /my/own/data/dir
```