import os

def nmt_train(args):
    """
    :param args:
    :return:
    """
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    path = uppath(__file__, 2)
    os.system("python {} --config_path {} --mode train".format(os.path.join(path, "main.py"), args.config_path))
