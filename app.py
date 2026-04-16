from flask import Flask, render_template, request
from model import predict_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        result = predict_text(text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
