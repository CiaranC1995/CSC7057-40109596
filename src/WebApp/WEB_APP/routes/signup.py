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
    
@signup_blueprint.route("/signup", methods=["POST"])
def signup_route_post():
    
    login_message = ""
    login_status = False

    endpoint1 = "http://127.0.0.1:8080/getAllUserInfo"
    api_response = requests.get(endpoint1)
    user_data = api_response.json()

    does_user_already_exist = False
    username = request.form["usernameField"]
    email = request.form["emailField"]
    password = request.form["passwordField"]

    for user in user_data:
        if user[1] == username or user[2] == email:
            does_user_already_exist = True
            break

    if does_user_already_exist:
        return render_template("signupMessage.html", doesUserAlreadyExist=does_user_already_exist, loginMessage=login_message, loginStatus=login_status)
    else:
        endpoint = "http://127.0.0.1:8080/signup"
        payload = {
            "username": username,
            "email": email,
            "password": password,
        }
        api_response = requests.post(endpoint, json=payload)
        return render_template("signupMessage.html", doesUserAlreadyExist=does_user_already_exist, loginMessage=login_message, loginStatus=login_status)