from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

# Create a blueprint for the route
deleteHistoryRoute_blueprint = Blueprint("deleteHistoryRoute", __name__)

# Initialize the database connector
connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)


@deleteHistoryRoute_blueprint.route("/deleteHistoryRoute", methods=["POST"])
def delete_history_route():
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id not provided"}), 400

        delete_sentence_perplexities_query = "DELETE sentence_perplexity FROM sentence_perplexity INNER JOIN classifier_output ON sentence_perplexity.classifier_output_id = classifier_output.classifier_output_id WHERE classifier_output.user_id = %s;"
        delete_classifier_outputs_query = (
            "DELETE FROM classifier_output WHERE classifier_output.user_id = %s;"
        )

        connector.execute_delete_query(delete_sentence_perplexities_query, (user_id,))
        connector.execute_delete_query(delete_classifier_outputs_query, (user_id,))

        return jsonify({"Message": "User History Deleted"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
