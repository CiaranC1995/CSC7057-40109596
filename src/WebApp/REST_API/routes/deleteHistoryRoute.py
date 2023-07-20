from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# Create a blueprint for the route
deleteHistoryRoute_blueprint = Blueprint("deleteHistoryRoute", __name__)


@deleteHistoryRoute_blueprint.route("/deleteHistoryRoute", methods=["POST"])
def delete_history_route():
    user_id = request.args.get("user_id")
    deleteAllSentencePerplexities = f"DELETE sentence_perplexity FROM sentence_perplexity INNER JOIN classifier_output ON sentence_perplexity.classifier_output_id = classifier_output.classifier_output_id WHERE classifier_output.user_id = {user_id};"
    deleteAllClassifierOutputs = (
        f"DELETE FROM classifier_output WHERE classifier_output.user_id = {user_id};"
    )

    try:
        connector.execute_delete_query(deleteAllSentencePerplexities)
        connector.execute_delete_query(deleteAllClassifierOutputs)

        return jsonify({"Message": "User History Deleted"})

    except Exception as e:
        return str(e)
