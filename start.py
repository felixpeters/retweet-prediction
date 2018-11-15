import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
hello = os.getenv("HELLO")
print("Value of environment variable HELLO: {}".format(hello))
