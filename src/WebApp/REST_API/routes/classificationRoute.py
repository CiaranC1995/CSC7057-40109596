from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

# Create a blueprint for the routes
classificationRoute_blueprint = Blueprint("classificationRoute", __name__)

# Initialize the database connector
connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

@classificationRoute_blueprint.route("/classificationRoute", methods=["POST"])
def classify():
    try:
        data = request.json
        required_fields = [
            "date_of_classification",
            "raw_input_text",
            "preprocessed_input_text",
            "text_perplexity",
            "text_burstiness",
            "classifier_output",
            "human_probability",
            "ai_probability",
            "user_id",
        ]

        for field in required_fields:
            if field not in data:
                return (
                    jsonify({"error": f"Missing '{field}' in the request body."}),
                    400,
                )

        date_of_classification = data["date_of_classification"]
        raw_text = data["raw_input_text"]
        preprocessed_text = data["preprocessed_input_text"]
        text_perplexity = data["text_perplexity"]
        text_burstiness = data["text_burstiness"]
        classifier_output = data["classifier_output"]
        human_probability = data["human_probability"]
        ai_probability = data["ai_probability"]
        user_id = data["user_id"]

        create_classification_output_query = (
            "INSERT INTO `classifier_output` "
            "(`classifier_output_id`, `date_of_classification`, `raw_input_text`, "
            "`preprocessed_input_text`, `text_perplexity`, `text_burstiness`, "
            "`classifier_output`, `human_probability`, `ai_probability`, `user_id`) "
            "VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        )

        values = (
            date_of_classification,
            raw_text,
            preprocessed_text,
            text_perplexity,
            text_burstiness,
            classifier_output,
            human_probability,
            ai_probability,
            user_id,
        )

        cursor = connector.execute_insert_query(
            create_classification_output_query, values
        )
        inserted_id = cursor.lastrowid

        return jsonify(
            {
                "message": "Database Record Created...",
                "classifier_output_id": inserted_id,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
