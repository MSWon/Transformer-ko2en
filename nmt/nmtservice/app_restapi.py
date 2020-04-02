from flask import Flask, request, render_template, jsonify
from nmt.translate import Translate
import yaml
import argparse
import os
import urllib

db_trans ={'고맙습니다': 'Thank you', 
           '감사합니다': 'Thank you', 
           '고마워': 'Thank you', 
           '안녕': 'Hello', 
           '안녕하세요': 'Hello', 
           '잘가': 'Goodbye', 
           '잘가요': 'Goodbye', 
           '바보': 'fool', 
           '멍청이': 'frat', 
           '슬퍼': 'sad', 
           '슬퍼요': 'sad', 
           '슬프군': "That's", 
           '아빠': 'Father', 
           '엄마': 'Mother', 
           '형': 'Brother', 
           '남동생': 'Brother', '여동생': 'Sister', '누나': 'Sister', '할머니': 'Grandma', '할아버지': 'Grandpa'}

uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
path = uppath(__file__, 2)

app = Flask(__name__)

@app.route('/nmt', methods=['GET'])
def index():
    input_sent = urllib.parse.unquote(request.args.get("text"))
    src_type = request.args.get("source")
    tgt_type = request.args.get("target")
    if input_sent in db_trans:
        output_sent = db_trans[input_sent]
    else:
        output_sent = model.service_infer(input_sent)
    return jsonify({"srcLangType":src_type, "tgtLangType":tgt_type, "translatedText":output_sent})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", help="port number", required=True)
    args = parser.parse_args()
    hyp_args = yaml.load(open(os.path.join(path, "train_config.yaml")))
    ## Build model
    model = Translate(hyp_args)
    app.run(host='0.0.0.0', port=args.port, debug=True)
