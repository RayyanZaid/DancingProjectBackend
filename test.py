import os

def make_directory(name : str):

    # if the directory does NOT exist, create it
    if not os.path.isdir(name):
        os.mkdir(name)


# call
make_directory("rayyan")