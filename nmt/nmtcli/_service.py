

def nmt_service(args):
    """
    :param args:
    :return:
    """
    import os

    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    path = uppath(__file__, 2)

    if args.mode == "restapi":
        code_path = os.path.join(path, "nmtservice/app_restapi.py")
    elif args.mode == "website":
        code_path = os.path.join(path, "nmtservice/app_website.py")
    else:
        raise NameError
    
    config_path = args.config_path
    port_num = args.port
    os.system(f"python {code_path} -c {config_path} -p {port_num}")
