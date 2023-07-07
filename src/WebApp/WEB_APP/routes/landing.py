from flask import render_template, Blueprint, session

landing_blueprint = Blueprint('landing', __name__)

@landing_blueprint.route('/')
def landing_route():
    sessionObject = session
    loginStatus = False

    if 'authen' in sessionObject:
        loginMessage = f"Logged In as '{sessionObject['user'][1]}'"
        loginStatus = True

        return render_template('landing.html', loginStatus=loginStatus, loginMessage=loginMessage)
    else:
        return render_template('landing.html', loginStatus=loginStatus)