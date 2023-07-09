from flask import Flask
from routes.landing import landing_blueprint
from routes.login import login_blueprint
from routes.logout import logout_blueprint
from routes.signup import signup_blueprint

app = Flask(__name__)
app.secret_key = 'thisismysecretkey1995'

# Web Application Blueprints
app.register_blueprint(login_blueprint)
app.register_blueprint(landing_blueprint)
app.register_blueprint(logout_blueprint)
app.register_blueprint(signup_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
