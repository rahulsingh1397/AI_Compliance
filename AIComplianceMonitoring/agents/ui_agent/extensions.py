from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_babel import Babel

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
babel = Babel()
