# from urllib import request
import re
import torch
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
import uuid
import json
from bson.objectid import ObjectId
from bson import json_util
import time
from chat_model_process import TransformerModel, PositionalEncoding, TokenEmbedding, eval
from blenderbot_process import blenderbot
model = torch.load(r"LangLern\PythonSimpleServer\model_music.pt")

# from aimod import blenderbot

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/LangLearn"
mongo = PyMongo(app)


@app.route("/registration", methods=["POST"])
def registration():
    # create new user and add to db + access(???)
    nickname = request.form["nickname"]
    password = request.form["password"]
    email = request.form["email"]

    if not nickname or not password or not email:
        return "not all fields are filled in", 400
        
    users_collection = mongo.db.users

    exist_nickname_user = users_collection.find_one({"nickname" : nickname})
    exist_email_user = users_collection.find_one({"email" : email})

    if exist_nickname_user: 
        return "this nickname is already used", 400
    elif exist_email_user:
         return "this email is already used", 400
    else:
        req = {"nickname" : nickname, "email" : email, "password": password}
        user = users_collection.insert_one(req).inserted_id
         
        token = str(uuid.uuid4())
        token_collection = mongo.db.tokens
        token_collection.insert_one({"user_id": str(user["_id"]), "token" : token}).inserted_id

    return token, 200


@app.route("/login", methods=["POST"])
def login():
    # access: create new token and add to db
    email = request.form["email"]
    password = request.form["password"]

    if not email or not password:
        return "not all fields are filled in", 400
    
    users_collection = mongo.db.users
    user = users_collection.find_one({"email" : email, "password": password})

    if not user:
        return "user does not exist", 400
    else:
        # create token
        token = str(uuid.uuid4())
        token_collection = mongo.db.tokens
        token_collection.insert_one({"user_id": str(user["_id"]), "token" : token}).inserted_id
        return token, 200


@app.route("/logout", methods=["GET"])
def logout():
    # delete token from db
    token = request.form["token"]
    tokens_collection = mongo.db.tokens
    tokens_collection.delete_one({"token" : token})
    return "user logout", 200


@app.route("/messagelist", methods=["POST"])
def messagelist():
    token = request.form["token"]
    theme = request.form["theme"]

    tokens_collection = mongo.db.tokens
    user_id = tokens_collection.find_one({"token" : token})

    if not user_id:
        return "user is not login", 400

    users_collection = mongo.db.users
    user = users_collection.find_one({"_id" : ObjectId(user_id["user_id"])})

    dialogs_collections = mongo.db.dialogs
    dialog = dialogs_collections.find_one({"user_id" : ObjectId(user["_id"]), "theme" : theme})

    if not dialog:
        return "empty dialog", 200

    messages_collection = mongo.db.messages
    dialog_history = messages_collection.find({"dialog_id" : ObjectId(dialog["_id"])})

    json_docs = [json.dumps(doc, default=json_util.default) for doc in dialog_history]

    return jsonify(json_docs), 200


@app.route("/message", methods=["POST"])
def message():
    token = request.form["token"]
    message = request.form["message"]
    theme = request.form["theme"]

    tokens_collection = mongo.db.tokens
    user_id = tokens_collection.find_one({"token" : token})["user_id"]

    users_collection = mongo.db.users
    user = users_collection.find_one({"_id" : ObjectId(user_id)})

    dialogs_collections = mongo.db.dialogs
    messages_collection = mongo.db.messages

    dialog = dialogs_collections.find_one({"user_id" : user["_id"], "theme" : theme})

    if not dialog:
        # create new dialog and new message with new dialog id
        dialogs_collections.insert_one({"user_id" : user["_id"], "theme" : theme})
        dialog = dialogs_collections.find_one({"user_id" : user["_id"], "theme" : theme})
        messages_collection.insert_one({"dialog_id" : ObjectId(dialog["_id"]), "sender" : "user", "message" : message})

    else:
        # add new message with existing dialog id
        messages_collection.insert_one({"dialog_id" : ObjectId(dialog["_id"]), "sender" : "user", "message" : message})
    
    # add neural network answer
    answer = blenderbot(message)
    # answer = eval(model, message)
    messages_collection.insert_one({"dialog_id" : ObjectId(dialog["_id"]), "sender" : "AI", "message" : answer})

    # updated dialog history
    dialog_history = messages_collection.find({"dialog_id" : ObjectId(dialog["_id"])})
    json_docs = [json.dumps(doc, default=json_util.default) for doc in dialog_history]

    return jsonify(json_docs), 200


if __name__ == "__main__":
  app.run(host="0.0.0.0")

