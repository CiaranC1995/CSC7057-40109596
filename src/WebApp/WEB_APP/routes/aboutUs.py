from flask import render_template, Blueprint, session, request
import requests

aboutUs_blueprint = Blueprint('aboutUs', __name__)

@aboutUs_blueprint.route('/aboutUs', methods=['GET'])
def aboutUs_route():
    if 'authen' in session:
        loginMessage = f"Logged In as '{session['user'][1]}'"
        loginStatus = True
        return render_template('aboutUs.html', sessionObject=session, loginStatus=loginStatus, loginMessage=loginMessage)
    else:
        loginStatus = False
        return render_template('aboutUs.html', loginStatus=loginStatus, sessionObject=session)