from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

# Create a blueprint for the routes
specificHistoryRoute_blueprint = Blueprint("specificHistoryRoute", __name__)

# Initialize the database connector
connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

@specificHistoryRoute_blueprint.route("/specificHistoryRoute", methods=["GET"])
def get_specific_history():
    try:
        classifier_output_id = request.args.get("classifier_output_id")
        if not classifier_output_id:
            return jsonify({"error": "classifier_output_id not provided"}), 400

        # Use parameterized queries to avoid SQL injection attacks
        query1 = "SELECT * FROM classifier_output WHERE classifier_output_id = %s;"
        query2 = "SELECT * FROM sentence_perplexity WHERE classifier_output_id = %s;"
        query3 = "SELECT * FROM sentence_perplexity WHERE classifier_output_id = %s ORDER BY perplexity DESC LIMIT 1;"

        # Execute the queries with the provided classifier_output_id
        data1 = connector.execute_query(query1, (classifier_output_id,))
        data2 = connector.execute_query(query2, (classifier_output_id,))
        data3 = connector.execute_query(query3, (classifier_output_id,))

        return jsonify(
            {
                "classifier_output": data1,
                "sentence_perplexity": data2,
                "largest_ppl": data3,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
