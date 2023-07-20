from flask import render_template, Blueprint, session, request, redirect, url_for
import requests

deleteSingleResult_blueprint = Blueprint('deleteSingleResult', __name__)

@deleteSingleResult_blueprint.route('/deleteSingleResult', methods=['POST'])
def delete_single_result():
    if 'authen' in session:
        loginMessage = f"Logged In as '{session['user'][1]}'"
        loginStatus = True
        isRecordDeleted = False
        classifier_output_id = request.form.get('classifier_output_id')
        endpoint = "http://127.0.0.1:8080/deleteSingleResultRoute"

        try:
            requests.post(endpoint, params={'classifier_output_id': classifier_output_id})
            isRecordDeleted = True
        except Exception as e: 
            return str(e)

        return redirect(url_for('history.history_route', loginStatus=loginStatus, loginMessage=loginMessage, isRecordDeleted=isRecordDeleted))
    else:
        loginStatus = False
        return render_template('landing.html', loginStatus=loginStatus)