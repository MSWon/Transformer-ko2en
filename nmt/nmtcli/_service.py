

def nmt_service(args):
    """
    :param args:
    :return:
    """
    import os
    import yaml
    from flask import Flask
    from nmt.nmtservice.service_transformer import ServiceTransformer

    app = Flask(__name__)

    if args.mode == "restapi":
        hyp_args = yaml.load(open(args.config_path))
        model = ServiceTransformer(hyp_args)
        from nmt.nmtservice.app_restapi import index
        app.run(host='0.0.0.0', port=args.port, debug=True)
    elif args.mode == "website":
        os.system("python {} --port {}".format(os.path.join(path, "nmtservice/app_website.py"), args.port))
    else:
        raise NameError
