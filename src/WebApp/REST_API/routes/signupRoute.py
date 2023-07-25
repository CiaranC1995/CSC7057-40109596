from flask import jsonify, Blueprint, request
from connection import DatabaseConnector
import bcrypt

# Create a blueprint for the routes
signupRoute_blueprint = Blueprint("signupRoute", __name__)

# Initialize the database connector
connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# Route for creating a user in the database
@signupRoute_blueprint.route("/signupRoute", methods=["POST"])
def signup():
    try:
        data = request.json
        required_fields = ['username', 'email', 'password']

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing '{field}' in the request body."}), 400

        username = data["username"]
        email = data["email"]
        password = data["password"]

        # Hash the password using bcrypt
        saltRounds = 10
        hashedPassword = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(saltRounds))

        createUser_query = "INSERT INTO user (user_id, username, email, password) VALUES (NULL, %s, %s, %s);"
        values = (username, email, hashedPassword)

        connector.execute_insert_query(createUser_query, values)

        return jsonify({"message": "User Created..."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Get User Info route
@signupRoute_blueprint.route('/getAllUserInfo', methods=['GET'])
def get_all_user_info():
    try:
        getUserInfo = "SELECT SQL_NO_CACHE * FROM user"
        data = connector.execute_query(getUserInfo)
        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
