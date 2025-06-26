from flask import Blueprint, render_template, redirect, url_for, flash, session, request, jsonify, current_app as app
from flask_login import login_user, logout_user, login_required, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta

from ..extensions import db
from ..forms import LoginForm, RegistrationForm
from ..config import Config

auth_bp = Blueprint('auth', __name__)
limiter = Limiter(key_func=get_remote_address)

# Rate limits
login_limiter = limiter.shared_limit("5 per minute", scope="auth")

@auth_bp.route('/login', methods=['GET', 'POST'])
@login_limiter
def login():
    from ..models import User, LoginAttempt
    try:
        if current_user.is_authenticated:
            app.logger.debug(f"User already authenticated: {current_user.username}")
            return redirect(url_for('dashboard.index'))
        
        form = LoginForm()
        lang = session.get('lang', 'en')
        ip = request.remote_addr
        user_agent = request.headers.get('User-Agent')
        
        if form.validate_on_submit():
            try:
                user = User.query.filter_by(username=form.username.data).first()
                
                # Track login attempt
                attempt = LoginAttempt(
                    username=form.username.data,
                    ip_address=ip,
                    user_agent=user_agent,
                    successful=False
                )
                
                password_match = False
                if user:
                    try:
                        password_match = check_password_hash(user.password, form.password.data)
                    except Exception as e:
                        app.logger.error(f"Password check error: {str(e)}")
                        raise
                
                app.logger.debug(f"Login attempt for user: {form.username.data}. User found: {bool(user)}. Password match: {password_match}")
                
                if user and password_match:
                    app.logger.debug(f"User {user.username} (ID: {user.id}) credentials valid. Calling login_user.")
                    login_user(user)
                    app.logger.debug(f"Called login_user. current_user.is_authenticated: {current_user.is_authenticated}")
                    session['lang'] = user.language
                    
                    # Generate JWT token
                    try:
                        token = jwt.encode({
                            'user_id': user.id,
                            'exp': datetime.utcnow() + timedelta(hours=1)
                        }, Config.SECRET_KEY, algorithm='HS256')
                        
                        attempt.successful = True
                        db.session.add(attempt)
                        db.session.commit()
                        
                        flash('Logged in successfully.', 'success')
                        app.logger.debug(f"Redirecting user {user.username} to dashboard.index.")
                        response = redirect(url_for('dashboard.index'))
                        response.set_cookie('access_token', token, httponly=True, secure=Config.SESSION_COOKIE_SECURE)
                        return response
                    except Exception as e:
                        app.logger.error(f"JWT token generation error: {str(e)}")
                        raise
                
                db.session.add(attempt)
                db.session.commit()
                app.logger.debug(f"Login unsuccessful for user: {form.username.data}.")
                flash('Login unsuccessful. Please check username and password.', 'danger')
            except Exception as e:
                app.logger.error(f"Login processing error: {str(e)}")
                db.session.rollback()
                flash('An error occurred during login. Please try again.', 'danger')
        elif form.errors:
            app.logger.debug(f"Login form validation errors: {form.errors}")

        return render_template('auth/login.html', form=form, lang=lang)
    except Exception as e:
        app.logger.error(f"Unexpected error in login route: {str(e)}")
        raise

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    from ..models import User
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    form = RegistrationForm()
    lang = session.get('lang', 'en')
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        user = User(username=form.username.data, email=form.email.data, 
                   password=hashed_password, role='user')
        db.session.add(user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('auth.login'))
    return render_template('auth/register.html', form=form, lang=lang)

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('lang', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/token/refresh', methods=['POST'])
@login_required
def refresh_token():
    new_token = jwt.encode({
        'user_id': current_user.id,
        'exp': datetime.utcnow() + timedelta(hours=1)
    }, Config.SECRET_KEY, algorithm='HS256')
    
    return jsonify({
        'access_token': new_token,
        'expires_in': 3600
    })
