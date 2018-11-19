import os
from subprocess import call
import boto3
from dotenv import find_dotenv, load_dotenv

# set constants
BUCKET = 'tep-research-project'
FILES = ['tweets.zip', 'transfer.zip']
DESTINATION = '/storage/'

# load credentials from environment variables
load_dotenv(find_dotenv())
ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")

# configure AWS client
client = boto3.client(
    's3',
    aws_access_key_id=KEY_ID,
    aws_secret_access_key=ACCESS_KEY,
)

# download files
for f in FILES:
    client.download_file(BUCKET, f, DESTINATION + f)

# unzip archives
call(["unzip", "/storage/*.zip", "-d", "/storage/"])
call(["ls", "-lh", "/storage/"])
