#! /bin/bash

wget -P /storage/ "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
mkdir /storage/glove
unzip /storage/glove.twitter.27B.zip -d /storage/glove/
