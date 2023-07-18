from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# Create a blueprint for the routes
sentencePerplexityRoute_blueprint = Blueprint("sentencePerplexityRoute", __name__)

@sentencePerplexityRoute_blueprint.route("/sentencePerplexityRoute", methods=["POST"])
def post_sentence_perplexity():
    sentence_number = request.json['sentence_number']
    sentence = request.json['sentence']
    perplexity = request.json['perplexity']
    classifier_output_id = request.json['classifier_output_id']

    createSentencePerplexityRecord = 'INSERT INTO sentence_perplexity (sentence_perplexity_id, sentence_number, sentence, perplexity, classifier_output_id) VALUES (NULL, %s, %s, %s, %s)'
    values = (
        sentence_number,
        sentence,
        perplexity,
        classifier_output_id
    )

    try:
        connector.execute_insert_query(createSentencePerplexityRecord, values)

        return jsonify(
            {
                "message": "Database Record Created...",
            }
        )
    except Exception as e:
        return str(e)