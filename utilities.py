from datetime import datetime


def write(data, filename, testinfo=""):
    f = open(filename, "a")
    now = datetime.now()
    time = now.strftime("%m,%d,%Y_%H,%M,%S")
    firstline = "info: " + str(testinfo) + " ; " + time + "\n"
    content = firstline + str(data)
    f.write(content)
    f.close()


def read(filename):
    f = open(filename)
    res = f.read()
    return res