from flask import render_template, Blueprint, session, request
import requests

specificHistory_blueprint = Blueprint('specificHistory', __name__)

@specificHistory_blueprint.route('/specificHistory', methods=['GET'])
def specificHistory_route():
    loginStatus = 'authen' in session
    loginMessage = f"Logged In as '{session.get('user', ['', ''])[1]}'" if loginStatus else ""

    if loginStatus:
        classifier_output_id = request.args.get('classifier_output_id')
        endpoint = f'http://127.0.0.1:8080/specificHistoryRoute'
        
        try:
            api_response = requests.get(endpoint, params={'classifier_output_id': classifier_output_id})
            specificHistoryInfo = api_response.json()
            return render_template('specificHistory.html', loginStatus=loginStatus, loginMessage=loginMessage, specificHistoryInfo=specificHistoryInfo)
        except requests.exceptions.RequestException as e:
            return f"Error occurred: {e}"

    else:
        return render_template('landing.html', loginStatus=loginStatus, loginMessage=loginMessage)
