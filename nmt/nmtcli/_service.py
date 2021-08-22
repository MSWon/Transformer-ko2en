

def nmt_service(args):
    """
    :param args:
    :return:
    """
    import os

    uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
    path = uppath(__file__, 2)

    config_path = args.config_path
    port_num = args.port

    if args.mode == "api":
        code_path = os.path.join(path, "nmtservice/app_restapi.py")
        cmd = f"python {code_path} -c {config_path} -p {port_num}"
    elif args.mode == "website":
        code_path = os.path.join(path, "nmtservice/app_website.py")
        cmd = f"python {code_path} -c {config_path} -p {port_num}"
    elif args.mode == "serving":
        cmd = f"gunicorn 'nmt.serving.api:create_app(\"{config_path}\")' -b 0.0.0.0:{port_num}"
    else:
        raise NameError
    
    os.system(cmd)
