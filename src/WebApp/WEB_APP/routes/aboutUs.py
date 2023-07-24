from flask import render_template, Blueprint, session

aboutUs_blueprint = Blueprint('aboutUs', __name__)

@aboutUs_blueprint.route('/aboutUs', methods=['GET'])
def aboutUs_route():
    loginStatus = 'authen' in session
    loginMessage = f"Logged In as '{session.get('user', ['', ''])[1]}'" if loginStatus else ""

    return render_template('aboutUs.html', loginStatus=loginStatus, loginMessage=loginMessage)
