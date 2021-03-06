# use latest Tensorflow CPU image as base
FROM tensorflow/tensorflow:latest-py3

# install required packages
RUN apt-get update 
RUN apt-get install tree
RUN apt-get install wget
RUN pip install boto3
RUN pip install -U textblob
RUN pip install awscli --upgrade
RUN pip install python-twitter
RUN pip install -U python-dotenv

# keep h5py from locking files which causes errors
ENV HDF5_USE_FILE_LOCKING FALSE

# expose port for Jupyter notebook
EXPOSE 8888
