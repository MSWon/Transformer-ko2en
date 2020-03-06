from flask import Flask, request, render_template
from translate import Translate
import yaml

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_sent = request.form["txtSource"]
        output_sent = model.service_infer(input_sent)
        return render_template("index.html", input_sent=input_sent, output_sent=output_sent)
    return render_template("index.html")

if __name__ == '__main__':
    hyp_args = yaml.load(open("./train_config.yaml"))
    ## Build model
    model = Translate(hyp_args)
    app.run(host='0.0.0.0', port=5000, debug=True)