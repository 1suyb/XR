from flask import Flask, request, jsonify, send_file, render_template
import Modules

app = Flask(__name__)

@app.route('/')
def home() :
    return render_template("index.html")

@app.route('/processing',methods=['POST'])
def processing() :
    print("startprocessing")
    '''
    injson - {text : text}
    outjson - {data = [{id : id, url : url, img: img}, {},{}]}
    '''
    if not request.is_json :
        raise ValueError("is not JSON")
    text = request.get_json().get('text')
    results = Modules.processing(text)
    return jsonify(results)

if __name__ == '__main__' :
    app.run(host='0.0.0.0',debug=True)