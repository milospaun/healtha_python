from flask import Flask, request, render_template, jsonify
from datetime import datetime
import mysql.connector
import hashlib
import sqlite3

app = Flask(__name__)
app.config.from_pyfile('config.py')


@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/api/login', methods=['POST'])
def login():
    db = mysql.connector.connect(
        host=app.config.get('DB_HOST'),
        user=app.config.get('DB_USERNAME'),
        passwd=app.config.get('DB_PASSWORD'),
        database=app.config.get('DB_DATABASE'),
    )
    dbc = db.cursor()

    json = request.get_json()

    h = hashlib.sha256()
    h.update(json['password'].encode())
    passhash = h.hexdigest()
    dbc.execute("SELECT * FROM users WHERE email=%s", (json['email'], ))
    result = dbc.fetchone()
    if not result:
        dbc.close()
        db.close()
        return jsonify({
            "status": 400,
            "message": "Korisnik ne postoji",
            "entity": None,
        })
    if result[4] != passhash:
        dbc.close()
        db.close()
        return jsonify({
            "status": 400,
            "message": "Pogresan password",
            "entity": None,
        })
    else:
        h = hashlib.sha512((datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'/'+result[3]).encode())
        token = h.hexdigest()
        print(result)
        user = {
            'id': result[0],
            'name': result[1],
            'lastname': result[2],
            'email': result[3],
            'role': result[5],
            'role_id': result[6],
            'created_at': result[7],
            'token': token
        }
        dbc.close()
        db.close()
        return jsonify({
            "status": 200,
            "message": "Uspesno logovanje",
            "entity": user,
        })


@app.route('/api/registration', methods=['POST', 'GET'])
def registration():

    db = mysql.connector.connect(
        host=app.config.get('DB_HOST'),
        user=app.config.get('DB_USERNAME'),
        passwd=app.config.get('DB_PASSWORD'),
        database=app.config.get('DB_DATABASE'),
    )
    dbc = db.cursor()

    json = request.get_json()

    # validacija
    dbc.execute("SELECT * FROM users WHERE email = %s", (json['email'],) )
    result = dbc.fetchone()
    if result:
        dbc.close()
        db.close()
        return jsonify({
            "status": 400,
            "message": "Korisnik vec postoji",
            "entity": None,
        })

    # registracija
    h = hashlib.sha256()
    h.update(json['password'].encode())
    passhash = h.hexdigest()
    dbc.execute("INSERT INTO users(name, lastname, email, password, role, role_id) VALUES (%s, %s, %s, %s, %s, %s)"
                , (json['name'], json['lastname'], json['email'], passhash, json['role'], 1))
    user_id_dict = {
        "id": dbc.lastrowid
    }
    db.commit()



    # zatvori konekciju
    dbc.close()
    db.close()
    return jsonify({
        "status": 200,
        "message": "Uspesno kreiranje korisnika",
        "entity": {
            "name": json['name'],
            "lastname": json['lastname'],
            "email": json['email'],
            "role": json['role'],
            **user_id_dict,
            "token": hashlib.sha512((datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'/'+json['email']).encode()).hexdigest(),
        },
    })



@app.route('/api/data', methods=['POST'])
def receive_data():
    db = mysql.connector.connect(
        host=app.config.get('DB_HOST'),
        user=app.config.get('DB_USERNAME'),
        passwd=app.config.get('DB_PASSWORD'),
        database=app.config.get('DB_DATABASE'),
    )
    dbc = db.cursor()

    try:
        json = request.get_json()
        print(json)
        # return jsonify({
        #     "json": json,
        # })
        data = json['data']
        data_array = []
        for row in data:
            data_array.append((*tuple(row.values()), json['user_id'], json['source'], datetime.now()))
        query = "INSERT INTO data(gx, gy, gz, ax, ay, az, timestamp, user_id, source, created_at) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        dbc.executemany(query, data_array)
        db.commit()


        dbc.close()
        db.close()
        return jsonify({
            "status": 200,
            "message": "Uspesno primljeni podaci"
        })
    except Exception as e:
        dbc.close()
        db.close()
        return jsonify({
            "status": 400,
            "message": str(e),
            "json": request.get_json()
        })



@app.route('/mgr', methods=['GET'])
def migrate():
    db = mysql.connector.connect(
        host=app.config.get('DB_HOST'),
        user=app.config.get('DB_USERNAME'),
        passwd=app.config.get('DB_PASSWORD'),
        database=app.config.get('DB_DATABASE'),
    )
    dbc = db.cursor()
    f = open('schema_mysql.sql', 'r')
    schema = f.read()
    queries = schema.split('#')
    for query in queries:
        dbc.execute(query)
    dbc.close()
    db.close()
    return jsonify({
        "status": '200',
        "message": "Uspesno migriranje",
    })

@app.route('/drp', methods=['GET'])
def drop_db():
    db = mysql.connector.connect(
        host=app.config.get('DB_HOST'),
        user=app.config.get('DB_USERNAME'),
        passwd=app.config.get('DB_PASSWORD'),
        database=app.config.get('DB_DATABASE'),
    )
    dbc = db.cursor()
    f = open('drop_schema.sql', 'r')
    schema = f.read()
    queries = schema.split('#')
    for query in queries:
        dbc.execute(query)
    dbc.close()
    db.close()
    return jsonify({
        "status": '200',
        "message": "Uspesno dropovanje baze",
    })







