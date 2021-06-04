import os
from setuptools import setup, find_packages
from nmt import __version__, __author__, __description__
from nmt.nmtcli.cli import main

def get_requires():
    result = []
    with open("./requirements.txt", "r") as f:
        for package_name in f:
            result.append(package_name)
    return result

setup(
    name='nmt',                      # 모듈명
    version=__version__,             # 버전
    author=__author__,               # 저자
    description=__description__,     # 설명
    packages=find_packages(),
    python_requires='>=3.6.0',
    install_requires=get_requires(), # 패키지 설치를 위한 요구사항 패키지들
    entry_points={
        # nmt라는 명령어를 실행하면
        # nmt.nmtcli모듈 cli.py에서 main함수를 실행한다는 의미
        "console_scripts" : ["nmt=nmt.nmtcli.cli:main"]
    },
    scripts=[
        'nmt/multi-bleu.perl', 'nmt/train_config.yaml',
        'nmt/nmtservice/templates/index.html', 'nmt/nmtservice/static/css/main.css'
    ],
    include_package_data=True,
    zip_safe=False,
)