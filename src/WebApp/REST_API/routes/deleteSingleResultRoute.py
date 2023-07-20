from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# Create a blueprint for the route
deleteSingleRecordRoute_blueprint = Blueprint("deleteSingleRecord", __name__)


@deleteSingleRecordRoute_blueprint.route("/deleteSingleRecord", methods=["POST"])
def delete_single_record():
    classifier_output_id = request.args.get("classifier_output_id")

    deleteSentencePerplexities = f"DELETE FROM sentence_perplexity WHERE classifier_output_id = {classifier_output_id};"
    deleteClassifierOutput = f"DELETE FROM classifier_output WHERE classifier_output_id = {classifier_output_id};"

    try:
        connector.execute_delete_query(deleteSentencePerplexities)
        connector.execute_delete_query(deleteClassifierOutput)

        return jsonify({"Message": "Record Deleted"})
    
    except Exception as e:
        return str(e)
