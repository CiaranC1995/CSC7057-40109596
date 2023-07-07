from flask import render_template, Blueprint

landing_blueprint = Blueprint('landing', __name__)

@landing_blueprint.route('/')
def landing_route():
    return render_template('landing.html')