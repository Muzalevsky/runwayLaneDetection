# lane-detection-medium

<div align="center">

[![PythonSupported](https://img.shields.io/badge/python-3.9-brightgreen.svg)](https://python3statement.org/#sections50-why)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Bandit](https://img.shields.io/badge/security-bandit-informational.svg)](https://github.com/PyCQA/bandit)

Awesome `lane-detection-medium` project!

</div>

- [Repository contents](#repository-contents)
- [System requirements](#system-requirements)
- [Other interesting info](#other-interesting-info)

## Repository contents

- [docs](docs) - documentation of the project
- [reports](reports) - reports generated (as generated from notebooks)
  > Check if you need to ignore large reports or keep them in Git LFS
- [configs](configs) - configuration files directory



- [scripts](scripts) - repository service scripts
  > These ones are not included into the pakckage if you build one - these scripts are only for usage with repository
- [lane_detection_medium](lane_detection_medium) - source files of the project
- [.editorconfig](.editorconfig) - configuration for [editorconfig](https://editorconfig.org/)
- [.flake8](.flake8) - [flake8](https://github.com/pycqa/flake8) linter configuration
- [.gitignore](.gitignore) - the files/folders `git` should ignore
- [.pre-commit-config.yaml](.pre-commit-config.yaml) - [pre-commit](https://pre-commit.com/) configuration file
- [README.md](README.md) - the one you read =)
- [DEVELOPMENT.md](DEVELOPMENT.md) - guide for development team
- [Makefile](Makefile) - targets for `make` command
- [cookiecutter-config-file.yml](cookiecutter-config-file.yml) - cookiecutter project config log
- [poetry.toml](poetry.toml) - poetry local config
- [pyproject.toml](pyproject.toml) - Python project configuration

## System requirements

- Python version: 3.9
- Operating system: Ubuntu 20.04
- Poetry version >= 1.2.0

> We tested on this setup - you can try other versions or operation systems by yourself!

## Other interesting info

Here you can write anything about your project!

And here is the result of the latest checkpoint:
![DemoGIF](demo/test_archangel.gif)

Metrics:
|             | Precision | Recall | mAP50 | mAP50-95 |
|-------------|-----------|--------|-------|----------|
| all         | 0.855     | 0.855  | 0.879 | 0.606    |
| solid white | 0.959     | 0.843  | 0.93  | 0.7      |
| break white | 0.751     | 0.867  | 0.828 | 0.512    |

