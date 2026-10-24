# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

First version, built from source so far. It runs ER, HIO, RAAR, shrinkwrap, detwinning and
multiple random starts on 2D and 3D far-field intensities, following QoraPhase operator for
operator. A CPU reference backend and a CUDA backend sit behind one C++ API, with a Python API
shaped like `qoraphase.reconstruct`.
