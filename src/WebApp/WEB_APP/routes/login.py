from flask import render_template, Blueprint, session, request
import requests

login_blueprint = Blueprint('login', __name__)

@login_blueprint.route('/login', methods=['GET'])
def login_route():
    if 'authen' in session:
        loginMessage = f"Already Logged In as '{session['user'][1]}'"
        loginStatus = True
        return render_template('landing.html', sessionObject=session, loginStatus=loginStatus, loginMessage=loginMessage)
    else:
        loginStatus = False
        return render_template('login.html', loginStatus=loginStatus, sessionObject=session)

@login_blueprint.route('/login', methods=['POST'])
def login_post_route():
    username = request.form.get('usernameField', False)
    password = request.form.get('passwordField', False)
    
    endpoint = 'http://127.0.0.1:8080/getAllUserInfo'

    try:
        api_response = requests.get(endpoint)
        api_response.raise_for_status() 

        user_info = api_response.json()
        for user in user_info:

            if username == user[1] and password == user[3]:
                session['user'] = user
                session['user_id'] = user[0]
                session['authen'] = True
                loginStatus = True
                loginMessage = f"Login Successful... Welcome Back {username}"
                return render_template('loginMessage.html', loginStatus=loginStatus, loginMessage=loginMessage)

        loginStatus = False
        loginMessage = 'Credentials Not Recognized... Please Try Again...'
        return render_template('loginMessage.html', loginStatus=loginStatus, loginMessage=loginMessage)

    except (requests.RequestException, ValueError) as e:
        error_message = f"Error occurred: {str(e)}"
        return render_template('error.html', error_message=error_message)
