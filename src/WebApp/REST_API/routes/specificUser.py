from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# Create a blueprint for the routes
specificUser_blueprint = Blueprint("specificUser", __name__)

@specificUser_blueprint.route("/specificUser", methods=["GET"])
def specific_user():

    user_id = request.json["user_id"]

    try:
        getSpecificUserInfo = f"SELECT SQL_NO_CACHE * FROM user WHERE user_id = {user_id}"
        data = connector.execute_query(getSpecificUserInfo)
        return jsonify(data)
    except Exception as e:
        return str(e)