from flask import Blueprint, render_template, redirect, url_for, flash, request, session, current_app
from flask_login import login_required, current_user
import bcrypt
import sys
import os

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions import db
from forms import ProfileForm, PasswordChangeForm

settings_bp = Blueprint('settings', __name__)

@settings_bp.route('/', methods=['GET', 'POST'])
@login_required
def index():
    lang = session.get('lang', current_user.language if current_user else 'en')
    profile_form = ProfileForm(obj=current_user)
    password_form = PasswordChangeForm()
    
    if request.method == 'POST':
        if 'update_profile' in request.form and profile_form.validate_on_submit():
            current_user.username = profile_form.username.data
            current_user.email = profile_form.email.data
            current_user.language = profile_form.language.data
            current_user.timezone = profile_form.timezone.data
            current_user.email_notifications = profile_form.email_notifications.data
            current_user.in_app_notifications = profile_form.in_app_notifications.data
            db.session.commit()
            session['lang'] = current_user.language
            flash('Profile updated successfully.', 'success')
            return redirect(url_for('settings.index'))
        
        elif 'change_password' in request.form and password_form.validate_on_submit():
            if bcrypt.checkpw(password_form.current_password.data.encode('utf-8'), 
                             current_user.password.encode('utf-8')):
                hashed_password = bcrypt.hashpw(password_form.new_password.data.encode('utf-8'), bcrypt.gensalt())
                current_user.password = hashed_password.decode('utf-8')
                db.session.commit()
                flash('Password changed successfully.', 'success')
                return redirect(url_for('settings.index'))
            flash('Current password is incorrect.', 'danger')
    
    return render_template('settings/index.html', 
                         profile_form=profile_form, 
                         password_form=password_form, 
                         lang=lang)

@settings_bp.route('/set_language/<lang>')
@login_required
def set_language(lang):
    if lang in current_app.config['LANGUAGES']:
        session['lang'] = lang
        if current_user.is_authenticated:
            current_user.language = lang
            db.session.commit()
    return redirect(request.referrer or url_for('dashboard.index'))
