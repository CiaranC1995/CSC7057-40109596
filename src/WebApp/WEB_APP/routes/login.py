from flask import render_template, Blueprint

login_blueprint = Blueprint('login', __name__)

@login_blueprint.route('/login')
def login_route():
    return render_template('login.html')