from flask import render_template, Blueprint, session, request
import requests

history_blueprint = Blueprint('history', __name__)

@history_blueprint.route('/history', methods=['GET'])
def history_route():
    if 'authen' in session:
        loginMessage = f"Logged In as '{session['user'][1]}'"
        loginStatus = True

        user_id = session['user_id']
        endpoint = 'http://127.0.0.1:8080/userResults'
        
        try:
            api_response = requests.get(endpoint, json={"user_id": user_id})
            history_info = api_response.json()
        except Exception as e:
            return str(e)

        return render_template('history.html', loginStatus=loginStatus, loginMessage=loginMessage, history_info=history_info)
    else:
        loginStatus = False
        return render_template('landing.html', loginStatus=loginStatus)