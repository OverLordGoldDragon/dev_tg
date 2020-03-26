#!/usr/bin/env bash
pycodestyle --max-line-length=89 --ignore=E221,E241,E225,E226,E402,E722,E741,E272,W503,W504 train_generatorr tests

pytest -s --cov=train_generatorr tests/
