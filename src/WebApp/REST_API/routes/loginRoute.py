from flask import jsonify, Blueprint
from connection import DatabaseConnector

connector = DatabaseConnector(host='localhost', user='root', password='root', database='csc7057')

# Create a blueprint for the routes
loginRoute_blueprint = Blueprint('loginRoute', __name__)

# Route for retrieving user info from the database
@loginRoute_blueprint.route('/getAllUserInfo', methods=['GET'])
def get_all_user_info():
    try:
        getUserInfo = "SELECT * FROM user"
        data = connector.execute_query(getUserInfo)
        return jsonify(data)
    except Exception as e:
        return str(e)