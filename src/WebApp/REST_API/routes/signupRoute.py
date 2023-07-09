from flask import jsonify, Blueprint, request
from connection import DatabaseConnector
import bcrypt

connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# Create a blueprint for the routes
signupRoute_blueprint = Blueprint("signupRoute", __name__)


# Route for retrieving user info from the database
@signupRoute_blueprint.route("/signup", methods=["POST"])
def signup():
    username = request.json["username"]
    email = request.json["email"]
    password = request.json["password"]

    saltRounds = 10
    hashedPassword = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(saltRounds))

    createUser = "INSERT INTO user (user_id, username, email, password) VALUES (NULL, %s, %s, %s)"
    values = (username, email, hashedPassword)

    try:
        connector.execute_insert_query(createUser, values)

        return jsonify(
            {
                "message": "User Created...",
            }
        )
    except Exception as e:
        return jsonify({
            'error': str(e)
        })

# Get User Info route 
@signupRoute_blueprint.route('/getAllUserInfo', methods=['GET'])
def get_all_user_info():
    try:
        getUserInfo = "SELECT * FROM user"
        data = connector.execute_query(getUserInfo)
        return jsonify(data)
    except Exception as e:
        return str(e)
