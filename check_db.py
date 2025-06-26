from sqlalchemy import create_engine, inspect
import sys

# Use the exact path confirmed from the Flask app's debug output
db_uri = 'sqlite:///E:\\Projects\\Private\\AI_Compliance\\instance\\app.db'

print(f"--- Database Schema Check ---")
print(f"Connecting to: {db_uri}")

try:
    engine = create_engine(db_uri)
    with engine.connect() as connection:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            print("\n[RESULT] FAILURE: No tables found in the database.")
            print("This indicates that the 'flask db upgrade' command did not run correctly or the migration is empty.")
        else:
            print("\n[RESULT] SUCCESS: Found the following tables:")
            for table in tables:
                print(f"  - {table}")
                
            
            if 'users' in tables:
                print("\n[VERDICT] The 'users' table EXISTS in the database file.")
            else:
                print("\n[VERDICT] The 'users' table DOES NOT EXIST in the database file.")

except Exception as e:
    print(f"\n[ERROR] An error occurred while trying to connect to the database: {e}", file=sys.stderr)

print("--- End of Check ---")