from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from summarize import summarize
from sen import sen
from ner import ner
from qna.main import main
import re

app = Flask(__name__)
CORS(app)

@app.route('/run-main', methods=['POST'])
@cross_origin()
def run_main():
    text = request.form.get('data1')
    text = re.sub('[^.,a-zA-Z0-9 \n\.]', '', text)
    summary = summarize(text)
    senn = sen(text)
    entity = ner(text)

    res = {}
    res['text'] = text
    res['summary'] = summary
    res['sen'] = float(senn)
    res['person'] = entity['person']
    res['org'] = entity['org']
    res['date'] = entity['date']
    res['loc'] = entity['loc']
    res['pdt'] = entity['pdt']
    res['event'] = entity['event']
    response = jsonify(res)
    print(res)
    return response

@app.route('/qna-main', methods=['POST'])
@cross_origin()
def run_qnamain():
    text = request.form.get('text')
    qn = request.form.get('qn')
    ans = main(text, qn)

    res = {}
    res['ans'] = ans
    response = jsonify(res)
    return response

app.run(debug=True, use_reloader=False)