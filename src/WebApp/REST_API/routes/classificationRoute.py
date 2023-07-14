from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# Create a blueprint for the routes
classificationRoute_blueprint = Blueprint("classificationRoute", __name__)


# Route for creating an entry in the database with the classifier output
@classificationRoute_blueprint.route("/classificationRoute", methods=["POST"])
def classify():
    raw_text = request.json["raw_input_text"]
    preprocessed_text = request.json["preprocessed_input_text"]
    text_perplexity = request.json["text_perplexity"]
    text_burstiness = request.json["text_burstiness"]
    classifier_output = request.json["classifier_output"]
    human_probability = request.json["human_probability"]
    ai_probability = request.json["ai_probability"]
    user_id = request.json["user_id"]

    createClassificationOutput = "INSERT INTO `classifier_output` (`classifier_output_id`, `raw_input_text`, `preprocessed_input_text`, `text_perplexity`, `text_burstiness`, `classifier_output`, `human_probability`, `ai_probability`, `user_id`) VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s)"
    values = (
        raw_text,
        preprocessed_text,
        text_perplexity,
        text_burstiness,
        classifier_output,
        human_probability,
        ai_probability,
        user_id,
    )

    try:
        connector.execute_insert_query(createClassificationOutput, values)

        return jsonify(
            {
                "message": "Database Record Created...",
            }
        )
    except Exception as e:
        return str(e)
