from flask import render_template, Blueprint, session, request
import requests

signup_blueprint = Blueprint('signup', __name__)

@signup_blueprint.route('/signup', methods=['GET'])
def signup_route():
    if 'authen' in session:
        loginMessage = f"Already Logged In as '{session['user'][1]}'"
        loginStatus = True
        return render_template('landing.html', sessionObject=session, loginStatus=loginStatus, loginMessage=loginMessage)
    else:
        loginStatus = False
        return render_template('signup.html', loginStatus=loginStatus, sessionObject=session)