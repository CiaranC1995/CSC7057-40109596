from flask import Flask
from connection import DatabaseConnector
from routes.loginRoute import loginRoute_blueprint
from routes.signupRoute import signupRoute_blueprint
from routes.contactRoute import contactRoute_blueprint
from routes.classificationRoute import classificationRoute_blueprint
from routes.specificUser import specificUser_blueprint
from routes.userResults import userResults_blueprint
from routes.sentencePerplexityRoute import sentencePerplexityRoute_blueprint
from routes.specificHistoryRoute import specificHistoryRoute_blueprint
from routes.deleteSingleResultRoute import deleteSingleRecordRoute_blueprint

app = Flask(__name__)

connector = DatabaseConnector(host='localhost', user='root', password='root', database='csc7057')

# Register API Blueprints
app.register_blueprint(loginRoute_blueprint)
app.register_blueprint(signupRoute_blueprint)
app.register_blueprint(contactRoute_blueprint)
app.register_blueprint(classificationRoute_blueprint)
app.register_blueprint(specificUser_blueprint)
app.register_blueprint(userResults_blueprint)
app.register_blueprint(sentencePerplexityRoute_blueprint)
app.register_blueprint(specificHistoryRoute_blueprint)
app.register_blueprint(deleteSingleRecordRoute_blueprint)

if __name__ == '__main__':
    app.run(port=8080, debug=True)