from flask import jsonify, Blueprint, request
from connection import DatabaseConnector

connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# Create a blueprint for the routes
contactRoute_blueprint = Blueprint("contactRoute", __name__)

# Route for creating an entry in the database with the contact message from the user 
@contactRoute_blueprint.route("/contactRoute", methods=["POST"])
def contact():
    name = request.json["name"]
    email = request.json["email"]
    subject = request.json["subject"]
    message = request.json["message"]

    createContactMessage = "INSERT INTO user_contact_message (user_contact_message_id, name, email, subject, message) VALUES (NULL, %s, %s, %s, %s)"
    values = (name, email, subject, message)

    try:
        connector.execute_insert_query(createContactMessage, values)

        return jsonify(
            {
                "message": "Database Record Created...",
            }
        )
    except Exception as e:
        return jsonify({
            'error': str(e)
        })