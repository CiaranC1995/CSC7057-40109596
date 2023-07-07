from flask import session, Blueprint, redirect, url_for

logout_blueprint = Blueprint('logout', __name__)

@logout_blueprint.route('/logout')
def logout_route():

    session.clear()

    return redirect(url_for('landing.landing_route'))