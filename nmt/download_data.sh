FILEID=1llEZdcALMJB8AFcJNTRsVpyZX3Brla4e
FILENAME=data.tar.gz
SCRIPT_PATH=$( cd "$(dirname "$0")" ; pwd )

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID'' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $SCRIPT_PATH/$FILENAME && rm -rf /tmp/cookies.txt

tar -zxvf $SCRIPT_PATH/$FILENAME -C $SCRIPT_PATH
rm $SCRIPT_PATH/$FILENAME
