from setuptools import setup, find_packages

setup(
    name='AIComplianceMonitoring',
    version='0.1.0',
    packages=find_packages(),
    description='A comprehensive suite for AI compliance monitoring, reporting, and integration.',
    author='AI Compliance Team',
    install_requires=[
        # Add your project's core dependencies here
        # For example: 'flask', 'sqlalchemy', etc.
        # The requirements from specific agents will be handled separately
    ],
)
