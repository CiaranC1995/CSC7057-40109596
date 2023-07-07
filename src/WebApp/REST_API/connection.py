import mysql.connector

class DatabaseConnector:

    def __init__(self, host, user, password, database):
        self.connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        # Check if the connection is established
        if self.connection.is_connected():
            print("Database connection established.")
        else:
            print("Database connection failed.")

    def execute_query(self, query):
        cursor = self.connection.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        return data