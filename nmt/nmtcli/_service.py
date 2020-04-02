import os

def nmt_service(args):
    """
    :param args:
    :return:
    """
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    path = uppath(__file__, 2)
    if args.mode == "restapi":
        os.system("python {} --port {}".format(os.path.join(path, "nmtservice/app_restapi.py"), args.port))
    elif args.mode == "website":
        os.system("python {} --port {}".format(os.path.join(path, "nmtservice/app_website.py"), args.port))
    else:
        raise NameError
