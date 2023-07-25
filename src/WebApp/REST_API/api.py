from flask import Flask
from connection import DatabaseConnector

app = Flask(__name__)

connector = DatabaseConnector(
    host="localhost", user="root", password="root", database="csc7057"
)

# List of Blueprints
BLUEPRINTS = [
    "loginRoute",
    "signupRoute",
    "contactRoute",
    "classificationRoute",
    "specificUser",
    "userResults",
    "sentencePerplexityRoute",
    "specificHistoryRoute",
    "deleteSingleResultRoute",
    "deleteHistoryRoute",
]

# Register Blueprints
for blueprint_name in BLUEPRINTS:
    blueprint_module = __import__(f"routes.{blueprint_name}", fromlist=[""])
    app.register_blueprint(getattr(blueprint_module, f"{blueprint_name}_blueprint"))

if __name__ == "__main__":
    app.run(port=8080, debug=True)
