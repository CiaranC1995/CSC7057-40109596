from flask import render_template, Blueprint, session, request
import requests

signup_blueprint = Blueprint('signup', __name__)

@signup_blueprint.route('/signup', methods=['GET', 'POST'])
def signup_route():
    loginStatus = 'authen' in session
    loginMessage = f"Already Logged In as '{session.get('user', ['', ''])[1]}'" if loginStatus else ""

    if loginStatus:
        return render_template('landing.html', loginStatus=loginStatus, loginMessage=loginMessage)
    else:
        if request.method == 'POST':
            endpoint1 = "http://127.0.0.1:8080/getAllUserInfo"
            api_response = requests.get(endpoint1)
            user_data = api_response.json()

            does_user_already_exist = any(user[1] == request.form["usernameField"] or user[2] == request.form["emailField"] for user in user_data)

            if does_user_already_exist:
                return render_template("signupMessage.html", doesUserAlreadyExist=does_user_already_exist, loginMessage=loginMessage, loginStatus=loginStatus)
            else:
                endpoint = "http://127.0.0.1:8080/signupRoute"
                payload = {
                    "username": request.form["usernameField"],
                    "email": request.form["emailField"],
                    "password": request.form["passwordField"],
                }
                api_response = requests.post(endpoint, json=payload)
                return render_template("signupMessage.html", doesUserAlreadyExist=False, loginMessage=loginMessage, loginStatus=loginStatus)

        return render_template('signup.html', loginStatus=loginStatus)
