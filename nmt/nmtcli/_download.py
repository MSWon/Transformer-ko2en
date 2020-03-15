import os


def nmt_download(args):
    """
    :param args:
    :return:
    """
    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    path = uppath(__file__, 2)

    if args.mode == "data":
        os.system("sh {}".format(os.path.join(path, "download_data.sh")))
    elif args.mode == "model":
        os.system("sh {}".format(os.path.join(path, "download_model.sh")))
    else:
        raise ValueError('mode should be (data/model)')
