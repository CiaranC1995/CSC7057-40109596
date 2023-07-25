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
    try:
        data = request.get_json()
        required_fields = ["name", "email", "subject", "message"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Field '{field}' is missing in the request data."}), 400

        name = data["name"]
        email = data["email"]
        subject = data["subject"]
        message = data["message"]

        createContactMessage = "INSERT INTO user_contact_message (user_contact_message_id, name, email, subject, message) VALUES (NULL, %s, %s, %s, %s)"
        values = (name, email, subject, message)

        connector.execute_insert_query(createContactMessage, values)

        return jsonify({"message": "Database Record Created..."}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
