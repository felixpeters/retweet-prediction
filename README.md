# Retweet Prediction Research Project

The aim of this project is to make accurate predictions of the eventual
retweet cascade size using NLP transfer learning approaches.

## Getting started

Tweet data for model training is not published yet, but AWS S3 storage bucket
will be opened for public access shortly.

This project is optimized for running on the [Paperspace Gradient](https://paperspace.com/gradient) platform,
but scripts are generally able to run in any environment containing the required
packages. See the accompanying `Dockerfile` for details about environment setup.
You might need to adjust paths for model and data storage.

## Structure

This repository is structured as follows:

* The root folder contains configuration files (Dockerfile, Makefile) and
scripts to reproduce experiments
* The `retpred` folder contains the corresponding module which implements
basic utilities for data processing and model training

## Contact

Contact me via [GitHub](https://github.com/felixpeters), [Twitter](https://twitter.com/_fpeters) or
[LinkedIn](https://www.linkedin.com/in/petersfelix/).
