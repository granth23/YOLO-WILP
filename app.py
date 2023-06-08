from flask import Flask, render_template, jsonify
from detect import analyze
from PIL import Image

app = Flask(__name__)
app.secret_key = "flask_key"

@app.route("/send-image/<path:url>")
def home(url):
    return jsonify(analyze(url))

if __name__ == '__main__':
    app.run(debug=True)