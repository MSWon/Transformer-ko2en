import os

def nmt_service(args):
    """
    :param args:
    :return:
    """
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    path = uppath(__file__, 2)
    os.system("python {} --port {}".format(os.path.join(path, "nmtservice/app.py") ,args.port))