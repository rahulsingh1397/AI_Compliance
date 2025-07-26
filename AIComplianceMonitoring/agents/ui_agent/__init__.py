# This file initializes the ui_agent package and makes its blueprints available.

from .scripts.main import main_bp
from .auth import auth_bp
from .dashboard import dashboard_bp
from .settings import settings_bp
from .api import api_bp