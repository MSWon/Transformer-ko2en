from flask import Flask, request, render_template
from translate import Translate
import yaml

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


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_sent = request.form["txtSource"]
        if input_sent in db_trans:
            output_sent = db_trans[input_sent]
        else:
            output_sent = model.service_infer(input_sent)
        return render_template("index.html", input_sent=input_sent, output_sent=output_sent)
    return render_template("index.html")

if __name__ == '__main__':
    hyp_args = yaml.load(open("./train_config.yaml"))
    ## Build model
    model = Translate(hyp_args)
    app.run(host='0.0.0.0', port=5000, debug=True)
