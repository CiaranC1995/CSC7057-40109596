from flask import render_template, Blueprint, session, request, redirect, url_for
import requests

deleteSingleResult_blueprint = Blueprint('deleteSingleResult', __name__)

@deleteSingleResult_blueprint.route('/deleteSingleResult', methods=['POST'])
def delete_single_result():
    loginStatus = 'authen' in session
    loginMessage = f"Logged In as '{session.get('user', ['', ''])[1]}'" if loginStatus else ""

    if loginStatus:
        classifier_output_id = request.form.get('classifier_output_id')
        endpoint = "http://127.0.0.1:8080/deleteSingleResultRoute"

        try:
            requests.post(endpoint, params={'classifier_output_id': classifier_output_id})
        except requests.exceptions.RequestException as e: 
            return f"An error occurred: {e}"

        return redirect(url_for('history.history_route', loginStatus=loginStatus, loginMessage=loginMessage))
    else:
        return render_template('landing.html', loginStatus=loginStatus)
