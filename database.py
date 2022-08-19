import sqlite3
import os
import sys
from sqlite3 import Error


class StateMachineDB:

    def __init__(self):

        # Check to see the DB directory exists
        if not os.path.exists(os.path.join(os.getcwd(), "database")):
            # Make the directories
            os.mkdir(os.path.join(os.getcwd(), "database"))
        # Database location
        db_location = os.path.join(os.getcwd(), "database", "db.sqlite")

        # Get the database name
        self.db_location = db_location
        # Get database
        self.db_conn = None

    def connect(self):
        # Establish connection
        try:
            self.db_conn = sqlite3.connect(self.db_location)
        except Error as e:
            sys.stdout.write(f"{e}\n")
        else:
            sys.stdout.write("Database Connection successful\n")

    def create_table(self, assembly_op):

        # Get the query
        query = "CREATE TABLE IF NOT EXISTS " + assembly_op + " "
        query += """
        (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        time_stamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        step_time TEXT NOT NULL,
        sequence_break TEXT,
        sequence_break_flag INT NOT NULL,
        missed_step TEXT,
        states_sequence TEXT NOT NULL
        )
        """

        # Create the table
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(query)
        except Error as e:
            sys.stdout.write(f"{e}\n")

    def insert_data(self, assembly_op, params):

        # Get the query
        query = "INSERT INTO " + assembly_op + " "
        query += """
        (step_time, sequence_break, sequence_break_flag, missed_step, states_sequence) VALUES 
        (?,?,?,?,?) 
        """

        try:
            cursor = self.db_conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid
        except Error as e:
            sys.stdout.write(f"{e}\n")

    def select_by_id(self, assembly_op, table_id):

        # Get query
        query = "SELECT * FROM " + assembly_op + " WHERE id = ?"

        try:
            cursor = self.db_conn.cursor()
            cursor.execute(query, (table_id, ))
            return cursor.fetchall()
        except Error as e:
            sys.stdout.write(f"{e}\n")
            return 0

    def query_last_rows(self, assembly_op, n=5):

        # Get query statement
        query = "SELECT * FROM " + assembly_op + " ORDER BY id DESC LIMIT " + str(n)

        # Execute
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(query)
            return cursor.fetchall()
        except Error as e:
            sys.stdout.write(f"{e}\n")
            return 0







