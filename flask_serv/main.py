from flask import Flask
from flask import request

from aimod import blenderbot
 
# Flask Constructor
app = Flask(__name__)
 
# decorator to associate a function with the url
@app.route("/")
def showHomePage():
    return "This is server request"

@app.route("/debug", methods=["POST"])
def debug():
    text = request.form["sample"]
    print(text)
    res = blenderbot(text)
    print(res)
    return res

@app.route("/login", methods=["POST"])
def login():
    text = request.form["email"]
    print(text)
    return "ok", 200
 
if __name__ == "__main__":
  app.run(host="0.0.0.0")