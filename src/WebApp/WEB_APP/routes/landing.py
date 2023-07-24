from flask import render_template, Blueprint, session

landing_blueprint = Blueprint('landing', __name__)

@landing_blueprint.route('/')
def landing_route():
    loginStatus = 'authen' in session
    loginMessage = f"Logged In as '{session.get('user', ['', ''])[1]}'" if loginStatus else ""

    return render_template('landing.html', loginStatus=loginStatus, loginMessage=loginMessage)
