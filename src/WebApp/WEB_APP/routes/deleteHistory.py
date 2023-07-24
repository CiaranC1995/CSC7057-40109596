from flask import render_template, Blueprint, session, request, redirect, url_for
import requests

deleteHistory_blueprint = Blueprint('deleteHistory', __name__)

@deleteHistory_blueprint.route('/deleteHistory', methods=['POST'])
def delete_history():
    loginStatus = 'authen' in session
    loginMessage = f"Logged In as '{session.get('user', ['', ''])[1]}'" if loginStatus else ""

    if loginStatus:
        user_id = session.get('user_id', None)
        endpoint = "http://127.0.0.1:8080/deleteHistoryRoute"

        if user_id is not None:
            try:
                requests.post(endpoint, params={'user_id': user_id})
            except requests.exceptions.RequestException as e: 
                return f"An error occurred: {e}"

        return redirect(url_for('history.history_route', loginStatus=loginStatus, loginMessage=loginMessage))
    else:
        return render_template('landing.html', loginStatus=loginStatus)
