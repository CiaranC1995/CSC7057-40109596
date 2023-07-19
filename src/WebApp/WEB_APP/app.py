from flask import Flask, session
from datetime import timedelta, datetime, timezone
from routes.landing import landing_blueprint
from routes.login import login_blueprint
from routes.logout import logout_blueprint
from routes.signup import signup_blueprint
from routes.contact import contact_blueprint
from routes.classification import classification_blueprint
from routes.history import history_blueprint
from routes.specificHistory import specificHistory_blueprint
from routes.aboutUs import aboutUs_blueprint

app = Flask(__name__)
app.secret_key = 'thisismysecretkey1995'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

MAX_INACTIVE_TIME = 3600

@app.before_request
def before_request():
    # Check if the session has expired
    last_active = session.get('last_active')
    if last_active is not None and (datetime.now(timezone.utc) - last_active).total_seconds() > MAX_INACTIVE_TIME:
        # Set a flag in the session to indicate the timeout
        session['session_timed_out'] = True

    # Update the last_active time in the session
    session['last_active'] = datetime.now(timezone.utc)

# Web Application Blueprints
app.register_blueprint(login_blueprint)
app.register_blueprint(landing_blueprint)
app.register_blueprint(logout_blueprint)
app.register_blueprint(signup_blueprint)
app.register_blueprint(contact_blueprint)
app.register_blueprint(classification_blueprint)
app.register_blueprint(history_blueprint)
app.register_blueprint(specificHistory_blueprint)
app.register_blueprint(aboutUs_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
