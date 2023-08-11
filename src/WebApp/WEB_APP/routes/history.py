from flask import render_template, Blueprint, session
import requests

history_blueprint = Blueprint('history', __name__)

@history_blueprint.route('/history', methods=['GET'])
def history_route():
    loginStatus = 'authen' in session
    loginMessage = f"Logged In as '{session.get('user', ['', ''])[1]}'" if loginStatus else ""

    if loginStatus:
        user_id = session.get('user_id')
        endpoint = 'http://127.0.0.1:8080/userResults'
        
        try:
            api_response = requests.get(endpoint, json={"user_id": user_id})
            history_info = api_response.json()

            if not history_info:
                empty_message = "No previous classification data available."
                return render_template('history.html', loginStatus=loginStatus, loginMessage=loginMessage, empty_message=empty_message, user_name=session['user'])

        except requests.exceptions.RequestException as e:
            return f"An error occurred: {e}"

        return render_template('history.html', loginStatus=loginStatus, loginMessage=loginMessage, history_info=history_info, user_name=session['user'])
    else:
        return render_template('landing.html', loginStatus=loginStatus)
