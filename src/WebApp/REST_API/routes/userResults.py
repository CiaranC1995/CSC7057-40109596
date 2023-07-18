from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# Create a blueprint for the routes
userResults_blueprint = Blueprint("userResults", __name__)


@userResults_blueprint.route("/userResults", methods=["GET"])
def user_results():
    user_id = request.json["user_id"]

    try:
        getUserResults = f"SELECT * FROM classifier_output WHERE user_id = {user_id};"
        data = connector.execute_query(getUserResults)
        return jsonify(data)
    except Exception as e:
        return str(e)
