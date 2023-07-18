from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# Create a blueprint for the routes
specificHistoryRoute_blueprint = Blueprint("specificHistoryRoute", __name__)


@specificHistoryRoute_blueprint.route("/specificHistoryRoute", methods=["GET"])
def get_specific_history():
    classifier_output_id = request.args.get("classifier_output_id")

    query1 = f"SELECT * FROM `classifier_output` WHERE classifier_output_id = {classifier_output_id};"
    query2 = f"SELECT * FROM sentence_perplexity WHERE classifier_output_id = {classifier_output_id};"

    try:
        data1 = connector.execute_query(query1)
        data2 = connector.execute_query(query2)
        return jsonify({"classifier_output": data1, "sentence_perplexity": data2})
    except Exception as e:
        return str(e)
