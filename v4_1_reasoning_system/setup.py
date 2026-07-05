"""Compatibility wrapper for legacy editable installs.

Prefer `pyproject.toml`; this file exists so `python setup.py develop`
still works on machines where modern wheel/build tooling is missing.
"""
from setuptools import setup


setup()
