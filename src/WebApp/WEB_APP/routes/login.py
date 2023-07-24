from flask import render_template, Blueprint, session, request
import requests
import bcrypt

login_blueprint = Blueprint('login', __name__)

@login_blueprint.route('/login', methods=['GET', 'POST'])
def login_route():
    loginStatus = 'authen' in session
    loginMessage = f"Logged In as '{session.get('user', ['', ''])[1]}'" if loginStatus else ""
    
    if loginStatus:
        return render_template('landing.html', loginStatus=loginStatus, loginMessage=loginMessage)
    else:
        if request.method == 'POST':
            username = request.form.get('usernameField', False)
            password = request.form.get('passwordField', False)

            endpoint = 'http://127.0.0.1:8080/getAllUserInfo'

            try:
                api_response = requests.get(endpoint)
                user_info = api_response.json()

                for user in user_info:
                    if username in [user[1], user[2]]:
                        is_match = bcrypt.checkpw(password.encode("utf-8"), user[3].encode("utf-8"))

                        if is_match:
                            session['user'] = user
                            session['user_id'] = user[0]
                            session['authen'] = True
                            session['session_timed_out'] = False

                            loginStatus = True
                            loginMessage = f"Login Successful... Welcome Back {username}"
                            return render_template('loginMessage.html', loginStatus=loginStatus, loginMessage=loginMessage)

                loginStatus = False
                loginMessage = 'Credentials Not Recognized... Please Try Again...'
                return render_template('loginMessage.html', loginStatus=loginStatus, loginMessage=loginMessage)

            except requests.exceptions.RequestException as e:
                return f"Error occurred: {e}"

        return render_template('login.html', loginStatus=loginStatus, sessionObject=session)
