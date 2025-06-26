from flask import Blueprint, redirect, url_for, jsonify

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return redirect(url_for('dashboard.index'))

@main_bp.route('/health')
def health_check():
    return jsonify({'status': 'ok', 'message': 'UI Agent operational'})
