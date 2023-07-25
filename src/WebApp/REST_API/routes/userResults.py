from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

# Create a blueprint for the routes
userResults_blueprint = Blueprint("userResults", __name__)

# Initialize the database connector
connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

@userResults_blueprint.route("/userResults", methods=["GET"])
def user_results():
    try:
        user_id = request.json.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id not provided"}), 400

        # Use parameterized query to avoid SQL injection attacks
        get_user_results_query = "SELECT * FROM classifier_output WHERE user_id = %s;"
        data = connector.execute_query(get_user_results_query, (user_id,))

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
