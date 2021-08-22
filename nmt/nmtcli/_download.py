import os


def nmt_download(args):
    """
    :param args:
    :return:
    """
    from nmt.download_model import download_file_from_google_drive
    if args.mode == "data":
        FILE_ID = "1llEZdcALMJB8AFcJNTRsVpyZX3Brla4e"
        FILE_NAME = "data.tar.gz"
        TOTAL_SIZE = 206_572_573
        download_file_from_google_drive(FILE_ID, FILE_NAME, TOTAL_SIZE)
    elif args.mode == "model":
        FILE_ID = "1A3Lt2U1l8xf4Ky2R-xfk75cXKDrPJvi_"
        FILE_NAME = "koen.2021.0704.tar.gz"
        TOTAL_SIZE = 346_964_756
        download_file_from_google_drive(FILE_ID, FILE_NAME, TOTAL_SIZE)
    else:
        raise ValueError('mode should be (data/model)')
