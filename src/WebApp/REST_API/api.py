from flask import Flask
from connection import DatabaseConnector
from routes.loginRoute import loginRoute_blueprint
from routes.signupRoute import signupRoute_blueprint
from routes.contactRoute import contactRoute_blueprint

app = Flask(__name__)

connector = DatabaseConnector(host='localhost', user='root', password='root', database='csc7057')

# Register API Blueprints
app.register_blueprint(loginRoute_blueprint)
app.register_blueprint(signupRoute_blueprint)
app.register_blueprint(contactRoute_blueprint)

if __name__ == '__main__':
    app.run(port=8080, debug=True)