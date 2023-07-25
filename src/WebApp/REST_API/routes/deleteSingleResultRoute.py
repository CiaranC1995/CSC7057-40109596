from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

# Create a blueprint for the route
deleteSingleResultRoute_blueprint = Blueprint("deleteSingleResult", __name__)

# Initialize the database connector
connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)


@deleteSingleResultRoute_blueprint.route("/deleteSingleResultRoute", methods=["POST"])
def delete_single_result_route():
    try:
        data = request.args
        classifier_output_id = data.get("classifier_output_id")
        if not classifier_output_id:
            return jsonify({"error": "classifier_output_id not provided"}), 400

        # Use parameterized queries to avoid SQL injection attacks
        delete_sentence_perplexities_query = (
            "DELETE FROM sentence_perplexity WHERE classifier_output_id = %s;"
        )
        delete_classifier_output_query = (
            "DELETE FROM classifier_output WHERE classifier_output_id = %s;"
        )

        # Execute the delete queries with the provided classifier_output_id
        connector.execute_delete_query(
            delete_sentence_perplexities_query, (classifier_output_id,)
        )
        connector.execute_delete_query(
            delete_classifier_output_query, (classifier_output_id,)
        )

        return jsonify({"Message": "Result Deleted"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
