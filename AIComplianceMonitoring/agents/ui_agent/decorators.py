from functools import wraps
from flask import redirect, url_for
from flask_login import current_user

def role_required(role):
    """Decorator to restrict access to a route to a specific role."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role != role:
                return redirect(url_for('auth.login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator
