import os

import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from datetime import datetime
import mysql.connector
import hashlib
import sqlite3

from keras.engine.saving import load_model

from signal_processing import process_raw_data

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
    dbc.execute("SELECT * FROM users WHERE email=%s", (json['email'],))
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
        h = hashlib.sha512((datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '/' + result[3]).encode())
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


@app.route('/api/registration', methods=['POST'])
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
    dbc.execute("SELECT * FROM users WHERE email = %s", (json['email'],))
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
    # create statistics entry
    query = "INSERT INTO statistics(user_id) VALUES (%s)"
    params = (dbc.lastrowid,)
    dbc.execute(query, params)
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
            "token": hashlib.sha512(
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '/' + json['email']).encode()).hexdigest(),
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

    # upisi podatke
    json = request.get_json()

    # debug
    # print(json)

    data = json['data']
    if len(data) == 0:
        return jsonify({
            "status": 400,
            "message": "Poslato 0 podataka"
        })
    data_array = []
    for row in data:
        data_array.append((*tuple(row.values()), json['user_id'], json['source'], datetime.now()))
    # print(data_array)
    query = "INSERT INTO data(gx, gy, gz, ax, ay, az, timestamp, user_id, source, created_at) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    try:
        dbc.executemany(query, data_array)
        db.commit()
    except Exception as e:
        print(e)
        print(format(e))
        dbc.close()
        db.close()
        return jsonify({
            "status": 400,
            "message": format(e)
        })




    dbc.close()
    db.close()
    return jsonify({
        "status": 200,
        "message": "Uspesno primljeni podaci"
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


@app.route('/api/getActivity', methods=['GET'])
def test_extra_s_mobile():
    cwd = os.getcwd()
    # home_dir = os.path.abspath(os.path.join(cwd, os.pardir))
    print(cwd)

    model_path = cwd + "/model/ExtraS/model.txt"
    data_path = cwd + "/data/"

    # acc_raw = np.loadtxt(data_path + 'acc_mobile_hodanje.csv')
    # gyro_raw = np.loadtxt(data_path + 'gyr_mobile_hodanje.csv')
    acc_raw = pd.read_csv(data_path + 'acc_exp01_user01.csv', sep=' ', header=None, names=['c0', 'c1', 'c2', 'c3'])
    gyro_raw = pd.read_csv(data_path + 'gyro_exp01_user01.csv', sep=' ', header=None, names=['c0', 'c1', 'c2', 'c3'])
    acc_raw = acc_raw[['c0', 'c1', 'c2']].values
    gyro_raw = gyro_raw[['c0', 'c1', 'c2']].values
    print(f"{acc_raw.shape}   {gyro_raw.shape}   {acc_raw[0]}   {gyro_raw[0]}")

    x = process_raw_data(acc_raw, gyro_raw)
    x = x[:478]
    x = x.T

    model = load_model(model_path)
    ycapa = model.predict(x)
    ycapa = ycapa.argmax(axis=1)
    print(ycapa)

    return str(ycapa)

@app.route('/api/get_activity', methods=['POST'])
def get_activity():
    db = mysql.connector.connect(
        host=app.config.get('DB_HOST'),
        user=app.config.get('DB_USERNAME'),
        passwd=app.config.get('DB_PASSWORD'),
        database=app.config.get('DB_DATABASE'),
    )
    dbc = db.cursor(dictionary=True)
    cwd = os.getcwd()
    model_path = cwd + "/model/ExtraS/model.txt"
    json = request.get_json()

    try:
        a = 5/0
        # get statistics from db
        query = 'select * from statistics where user_id = %s'
        params = (json['user_id'],)
        dbc.execute(query, params)
        statistics = dbc.fetchone()
        # print('Statistics: ', statistics)

        # get data from db
        query = "SELECT * FROM data WHERE user_id = %s AND processed = 0"
        params = (json['user_id'],)
        dbc.execute(query, params)
        data = dbc.fetchall()
        skip_processing = False
        if len(data) < 478:
            skip_processing = True

        if not skip_processing:
            data_lastrowid = data[len(data) - 1]['id']
            print('data last row id: ', data_lastrowid)
            seconds_processed = dbc.rowcount * 0.020

            acc_data = []
            gyr_data = []
            timestamp = 0.020
            valid_data = 0
            last_row = {}
            for row in data:
                # if row['ax'] + row['ay'] + row['az'] != 0.0 \
                #         and row['gx'] + row['gy'] + row['gz'] != 0.0 != 0.0:
                #     acc_data.append([row['ax'], row['ay'], row['az'], "{:3.3f}".format(timestamp)])
                #     gyr_data.append([row['gx'], row['gy'], row['gz'], "{:3.3f}".format(timestamp)])
                #     timestamp += 0.020
                acc_zeros = 0
                gyr_zeros = 0
                if row['ax'] == 0.0:
                    acc_zeros += 1
                if row['ay'] == 0.0:
                    acc_zeros += 1
                if row['az'] == 0.0:
                    acc_zeros += 1
                if row['gx'] == 0.0:
                    gyr_zeros += 1
                if row['gy'] == 0.0:
                    gyr_zeros += 1
                if row['gz'] == 0.0:
                    gyr_zeros += 1

                if acc_zeros < 2 and gyr_zeros < 2:
                    if valid_data > 0:
                        if last_row['ax'] == row['ax']:
                            row['ax'] += 0.000010
                        if last_row['ay'] == row['ay']:
                            row['ay'] += 0.000010
                        if last_row['az'] == row['az']:
                            row['az'] += 0.000010
                        if last_row['gx'] == row['gx']:
                            row['gx'] += 0.000010
                        if last_row['gy'] == row['gy']:
                            row['gy'] += 0.000010
                        if last_row['gz'] == row['gz']:
                            row['gz'] += 0.000010
                    last_row = dict(row)
                    acc_data.append(
                        ["{:3.6f}".format(row['ax']), "{:3.6f}".format(row['ay']), "{:3.6f}".format(row['az']),
                         "{:3.3f}".format(timestamp)])
                    gyr_data.append(
                        ["{:3.6f}".format(row['gx']), "{:3.6f}".format(row['gy']), "{:3.6f}".format(row['gz']),
                         "{:3.3f}".format(timestamp)])
                    timestamp += 0.020
                    valid_data += 1


            print('valid data: ', valid_data)
            if valid_data < 478:
                print('not enough valid data for new processing')
                activity_dict = {
                    "walking": statistics["walking"],
                    "walking_upstairs": statistics["walking_upstairs"],
                    "walking_downstairs": statistics["walking_downstairs"],
                    "sitting": statistics["sitting"],
                    "standing": statistics["standing"],
                    "laying": statistics["laying"],
                    "stand_to_sit": statistics["stand_to_sit"],
                    "sit_to_stand": statistics["sit_to_stand"],
                    "sit_to_lie": statistics["sit_to_lie"],
                    "lie_to_sit": statistics["lie_to_sit"],
                    "stand_to_lie": statistics["stand_to_lie"],
                    "lie_to_stand": statistics["lie_to_stand"],
                    "total": statistics["total"],
                }

                # close db connection
                dbc.close()
                db.close()

                return jsonify(activity_dict)

            # generate csvs
            acc_filename = cwd + '/acc_temp_' + str(json['user_id']) + '.csv'
            gyr_filename = cwd + '/gyr_temp_' + str(json['user_id']) + '.csv'
            acc_df = pd.DataFrame(acc_data)
            gyr_df = pd.DataFrame(gyr_data)
            acc_df.to_csv(acc_filename, sep=' ', header=None, index=False)
            gyr_df.to_csv(gyr_filename, sep=' ', header=None, index=False)

            # return 'test'

            # predict activity
            acc_raw = pd.read_csv(acc_filename, sep=' ', header=None, names=['c0', 'c1', 'c2', 'c3'])
            gyro_raw = pd.read_csv(gyr_filename, sep=' ', header=None, names=['c0', 'c1', 'c2', 'c3'])
            acc_raw = acc_raw[['c0', 'c1', 'c2']].values
            gyro_raw = gyro_raw[['c0', 'c1', 'c2']].values

            x = process_raw_data(acc_raw, gyro_raw)
            x = x[:478]
            x = x.T

            model = load_model(model_path)
            ycapa = model.predict(x)
            ycapa = ycapa.argmax(axis=1)
            print(ycapa)

            # remove generated csvs as they're no longer needed
            os.remove(acc_filename)
            os.remove(gyr_filename)

            # process result
            size = ycapa.size
            activity_dict = {
                "walking": 0,
                "walking_upstairs": 0,
                "walking_downstairs": 0,
                "sitting": 0,
                "standing": 0,
                "laying": 0,
                "stand_to_sit": 0,
                "sit_to_stand": 0,
                "sit_to_lie": 0,
                "lie_to_sit": 0,
                "stand_to_lie": 0,
                "lie_to_stand": 0,
                "total": 0,
            }

            counter = 1
            for key in activity_dict:
                count = np.count_nonzero(ycapa == counter)
                # print(key + ': ' + str(count))
                activity_percentage = float(count) / float(size)
                activity_dict[key] = round(activity_percentage * seconds_processed) + statistics[key]
                counter += 1

            activity_dict["total"] = statistics["total"] + round(seconds_processed)
            # print(activity_dict)

            # save new statistics
            query = "UPDATE statistics SET walking = %s,walking_upstairs = %s,walking_downstairs = %s,sitting= %s ,standing = %s,laying = %s" \
                    ",stand_to_sit=%s,sit_to_stand=%s,sit_to_lie=%s,lie_to_sit=%s,stand_to_lie=%s,lie_to_stand=%s,total=%s WHERE user_id = %s"
            params = (activity_dict['walking'], activity_dict['walking_upstairs'], activity_dict['walking_downstairs'],
                      activity_dict['sitting']
                      , activity_dict['standing'], activity_dict['laying'], activity_dict['stand_to_sit'],
                      activity_dict['sit_to_stand']
                      , activity_dict['sit_to_lie'], activity_dict['lie_to_sit'], activity_dict['stand_to_lie'],
                      activity_dict['lie_to_stand']
                      , activity_dict['total'], json['user_id'])
            dbc.execute(query, params)
            db.commit()

            # delete processed data
            query = "DELETE FROM data WHERE user_id = %s AND id <= %s"
            params = (json['user_id'], data_lastrowid,)
            dbc.execute(query, params)
            db.commit()

            # close db connection
            dbc.close()
            db.close()

            return jsonify(activity_dict)
        else:
            print('not enough raw data for processing')
            activity_dict = {
                "walking": statistics["walking"],
                "walking_upstairs": statistics["walking_upstairs"],
                "walking_downstairs": statistics["walking_downstairs"],
                "sitting": statistics["sitting"],
                "standing": statistics["standing"],
                "laying": statistics["laying"],
                "stand_to_sit": statistics["stand_to_sit"],
                "sit_to_stand": statistics["sit_to_stand"],
                "sit_to_lie": statistics["sit_to_lie"],
                "lie_to_sit": statistics["lie_to_sit"],
                "stand_to_lie": statistics["stand_to_lie"],
                "lie_to_stand": statistics["lie_to_stand"],
                "total": statistics["total"],
            }

            # close db connection
            dbc.close()
            db.close()

            return jsonify(activity_dict)
    except Exception as e:
        # close db connection
        dbc.close()
        db.close()

        print(str(e))

        raise
        return jsonify({
            "status": 400,
            "message": str(e),
        })
