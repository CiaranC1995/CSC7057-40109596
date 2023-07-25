from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

# Create a blueprint for the routes
specificUser_blueprint = Blueprint("specificUser", __name__)

# Initialize the database connector
connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

@specificUser_blueprint.route("/specificUser", methods=["GET"])
def specific_user():
    try:
        user_id = request.json.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id not provided"}), 400

        # Use parameterized query to avoid SQL injection attacks
        get_specific_user_info_query = "SELECT SQL_NO_CACHE * FROM user WHERE user_id = %s;"
        data = connector.execute_query(get_specific_user_info_query, (user_id,))

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
