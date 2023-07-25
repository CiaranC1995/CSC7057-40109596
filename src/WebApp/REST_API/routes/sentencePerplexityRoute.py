from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

# Create a blueprint for the routes
sentencePerplexityRoute_blueprint = Blueprint("sentencePerplexityRoute", __name__)

# Initialize the database connector
connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

@sentencePerplexityRoute_blueprint.route("/sentencePerplexityRoute", methods=["POST"])
def post_sentence_perplexity():
    try:
        # Validate JSON payload and required fields
        data = request.json
        required_fields = [
            "sentence_number",
            "sentence",
            "perplexity",
            "classifier_output_id",
        ]

        for field in required_fields:
            if field not in data:
                return (
                    jsonify({"error": f"Missing '{field}' in the request body."}),
                    400,
                )

        # Extract data from the JSON payload
        sentence_number = data["sentence_number"]
        sentence = data["sentence"]
        perplexity = data["perplexity"]
        classifier_output_id = data["classifier_output_id"]

        # Use parameterized query to prevent SQL injection
        create_sentence_perplexity_record_query = "INSERT INTO sentence_perplexity (sentence_perplexity_id, sentence_number, sentence, perplexity, classifier_output_id) VALUES (NULL, %s, %s, %s, %s);"
        values = (sentence_number, sentence, perplexity, classifier_output_id)

        # Execute the INSERT query with the provided values
        connector.execute_insert_query(create_sentence_perplexity_record_query, values)

        return jsonify({"message": "Database Record Created..."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
