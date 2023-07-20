from flask import render_template, Blueprint, session, request, redirect, url_for
import requests

deleteHistory_blueprint = Blueprint('deleteHistory', __name__)

@deleteHistory_blueprint.route('/deleteHistory', methods=['POST'])
def delete_history():
    if 'authen' in session:
        loginMessage = f"Logged In as '{session['user'][1]}'"
        loginStatus = True
        user_id = session['user_id']
        endpoint = "http://127.0.0.1:8080/deleteHistoryRoute"

        try:
            requests.post(endpoint, params={'user_id': user_id})
        except Exception as e: 
            return str(e)

        return redirect(url_for('history.history_route', loginStatus=loginStatus, loginMessage=loginMessage))
    else:
        loginStatus = False
        return render_template('landing.html', loginStatus=loginStatus)