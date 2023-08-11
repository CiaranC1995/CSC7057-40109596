from flask import Flask, session
from datetime import datetime, timezone
from config import MAX_SESSION_IDLE_TIME, SECRET_KEY, PERMANENT_SESSION_LIFETIME

# Web Application Setup
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['PERMANENT_SESSION_LIFETIME'] = PERMANENT_SESSION_LIFETIME

# List of Blueprints
BLUEPRINTS = [
    'login',
    'landing',
    'logout',
    'signup',
    'contact',
    'classification',
    'history',
    'specificHistory',
    'aboutUs',
    'deleteSingleResult',
    'deleteHistory',
    'downloadPDF'
]

# Register Blueprints
for blueprint_name in BLUEPRINTS:
    blueprint_module = __import__(f'routes.{blueprint_name}', fromlist=[''])
    app.register_blueprint(getattr(blueprint_module, f'{blueprint_name}_blueprint'))

@app.before_request
def before_request():
    # Check if the session has expired
    last_active = session.get('last_active')
    if last_active is not None and (datetime.now(timezone.utc) - last_active).total_seconds() > MAX_SESSION_IDLE_TIME:
        # Set a flag in the session to indicate the timeout
        session['session_timed_out'] = True

    # Update the last_active time in the session
    session['last_active'] = datetime.now(timezone.utc)

if __name__ == '__main__':
    app.run(debug=True)
