# use latest Tensorflow CPU image as base
FROM tensorflow/tensorflow:latest-py3

# install required packages
RUN pip install boto3
RUN pip install python-twitter
RUN pip install -U python-dotenv

# expose port for Jupyter notebook
EXPOSE 8888
