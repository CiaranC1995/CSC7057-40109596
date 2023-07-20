from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# Create a blueprint for the route
deleteSingleResultRoute_blueprint = Blueprint("deleteSingleResult", __name__)


@deleteSingleResultRoute_blueprint.route("/deleteSingleResultRoute", methods=["POST"])
def delete_single_result_route():
    classifier_output_id = request.args.get("classifier_output_id")
    print(classifier_output_id)
    deleteSentencePerplexities = f"DELETE FROM sentence_perplexity WHERE classifier_output_id = {classifier_output_id};"
    deleteClassifierOutput = f"DELETE FROM classifier_output WHERE classifier_output_id = {classifier_output_id};"

    try:
        connector.execute_delete_query(deleteSentencePerplexities)
        connector.execute_delete_query(deleteClassifierOutput)

        return jsonify({"Message": "Result Deleted"})
    
    except Exception as e:
        return str(e)
