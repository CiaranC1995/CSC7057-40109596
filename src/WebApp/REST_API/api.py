from flask import Flask, jsonify
from connection import DatabaseConnector

app = Flask(__name__)

connector = DatabaseConnector(host='localhost', user='root', password='root', database='csc7057')

# Route for retrieving user info from the database
@app.route('/getAllUserInfo', methods=['GET'])
def login():

    try:
        query = "SELECT * FROM user"
        data = connector.execute_query(query)
        return jsonify(data)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(port=8080, debug=True)