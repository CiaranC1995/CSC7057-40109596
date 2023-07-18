from flask import render_template, Blueprint, session, request
import requests

specificHistory_blueprint = Blueprint('specificHistory', __name__)

@specificHistory_blueprint.route('/specificHistory', methods=['GET'])
def specificHistory_route():
    if 'authen' in session:
        loginMessage = f"Logged In as '{session['user'][1]}'"
        loginStatus = True

        # need an API route to pull all sentences, perplexities and classifier outputs from database, which will then be passed to template
        classifier_output_id = request.args.get('classifier_output_id')
        endpoint = f'http://127.0.0.1:8080/specificHistoryRoute'
        try:
            api_response = requests.get(endpoint, params={'classifier_output_id': classifier_output_id})
            specificHistoryInfo = api_response.json()
        except Exception as e:
            return str(e)

        return render_template('specificHistory.html', loginStatus=loginStatus, loginMessage=loginMessage, specificHistoryInfo=specificHistoryInfo)
    else:
        loginStatus = False
        return render_template('landing.html', loginStatus=loginStatus)