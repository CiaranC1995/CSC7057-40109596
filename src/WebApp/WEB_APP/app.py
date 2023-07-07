from flask import Flask
from routes.landing import landing_blueprint
from routes.login import login_blueprint

app = Flask(__name__)

app.register_blueprint(login_blueprint)
app.register_blueprint(landing_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
