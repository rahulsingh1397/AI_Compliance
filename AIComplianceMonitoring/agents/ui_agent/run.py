import logging
from AIComplianceMonitoring.agents.ui_agent.app import create_app

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('flask_debug.log', mode='w')
    ]
)

# Set all relevant loggers to DEBUG
loggers = [
    'werkzeug',
    'flask.app',
    'flask_login',
    'flask_wtf',
    'flask_sqlalchemy'
]

for logger_name in loggers:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

app = create_app()

app.logger.info("Starting AI Compliance Monitoring UI Agent with enhanced logging")

if __name__ == "__main__":
    # Show all errors directly in the browser during development
    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = True
    
    # Run with extra debug options
    app.run(
        debug=True,
        use_debugger=True,
        use_reloader=True,
        passthrough_errors=True,
        host='0.0.0.0',
        port=5000
    )
