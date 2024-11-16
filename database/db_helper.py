import sqlite3
from datetime import datetime, date, timedelta
import json
import os
import sys
from inspect import getsourcefile
# import mysql.connector

project_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
# print(project_dir)

sys.path.append(project_dir)
# print(sys.path)


class Database:
    def __init__(self):
        with open(os.path.join(project_dir, "database/db_config.json"), "r") as f:
            self.table_schemas = json.load(f)
        self.connection = sqlite3.connect(os.path.join(project_dir, "database/data.db"))
        self.cursor = self.connection.cursor()
        self.initialize_db()

    def initialize_db(self):
        self.create_all_tables()
        # self.populate_tables()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.connection.commit()
        self.connection.close()

    def create_all_tables(self):
        for table_name in self.table_schemas:
            if table_name == "pdf_status":
                self.create_table_foreign_key(table_name,"user_id","user_info")
            else:
                self.create_table(table_name)

    def create_table(self, table_name):
        if table_name in self.table_schemas:
            create_table_string = (
                "CREATE TABLE IF NOT EXISTS "
                + table_name
                + "("
                + ", ".join(self.table_schemas[table_name])
                + ");"
            )
            try:
                self.cursor.execute(create_table_string)
            except Exception as e:
                print("Exception:", e)

    def create_table_foreign_key(self, table_name,reference_col,reference_table):
        if table_name in self.table_schemas:
            create_table_string = (
                "CREATE TABLE IF NOT EXISTS "
                + table_name
                + "("
                + ", ".join(self.table_schemas[table_name])
                + ",FOREIGN KEY ("+ reference_col + ") REFERENCES " + reference_table + "("+reference_col+")"
                + ");"
            )
            try:
                self.cursor.execute(create_table_string)
            except Exception as e:
                print("Exception:", e)

    def add_record(self, table_name, info, ignore_first_col=True):
        # print(info)
        table_cols = [col.split()[0] for col in self.table_schemas[table_name]]
        if ignore_first_col:
            table_cols = table_cols[1:]

        insert_string = (
            "INSERT INTO "
            + table_name
            + " ("
            + ", ".join(table_cols)
            + ") VALUES ("
        )

        vals = []

        for col in table_cols:
            if isinstance(info[col], str) or isinstance(info[col], date):
                vals.append("'" + str(info[col]) + "'")
            elif isinstance(info[col], dict) or isinstance(info[col],list):
                vals.append("'" + json.dumps(info[col]) + "'")
            else:
                vals.append(str(info[col]))

        insert_string += ", ".join(vals) + ");"
        # print(insert_string)
        self.cursor.execute(insert_string)
        self.connection.commit()

    def update_record(self, table_name, match_vals, updated_vals):
        # table_cols = [col.split()[0] for col in self.table_schemas[table_name]]

        insert_string = "update " + table_name + " set "

        vals = []
        for col in updated_vals:
            if isinstance(updated_vals[col], str) or isinstance(updated_vals[col], date):
                vals.append(col + "=" + "'" + str(updated_vals[col]) + "'")
            elif isinstance(updated_vals[col],dict) or isinstance(updated_vals[col],list):
                vals.append(col+"="+"'"+json.dumps(updated_vals[col])+"'" )
            else:
                vals.append(col + "=" + str(updated_vals[col]))
        check_vals = []

        for col in match_vals:
            check_vals.append(col + "=" + "'" + str(match_vals[col]) + "'")

        # print(id_col_name, id_val)
        insert_string += ", ".join(vals) + " where " + " and ".join(check_vals) + ";"
        # print(insert_string)

        self.cursor.execute(insert_string)
        self.connection.commit()

    def get_table_data(self, table_name, match_vals=None,match_vals_filter=None):
        table_cols = [col.split()[0] for col in self.table_schemas[table_name]]

        select_string = "select " + ", ".join(table_cols) + " from " + table_name
        check_vals = []
        if match_vals:
            check_vals = []
            for col in match_vals:
                check_vals.append(col + "=" + "'" + str(match_vals[col]) + "'")

        if match_vals_filter:
            check_vals = []
            print(match_vals_filter)
            for col in match_vals_filter:
                check_vals.append(col + " " + match_vals_filter[col][0] + " " + "'" + str(match_vals_filter[col][1]) + "'")

        if len(check_vals):
            select_string += " where " + " and ".join(check_vals)

        select_string += ";"
        # print(select_string)
        self.cursor.execute(select_string)

        data = []
        for row in self.cursor.fetchall():
            data.append({k: v for k, v in zip(table_cols, row)})

        # print(data)

        return data

    def get_data_with_conditions(self, table_name, conditions=None):
        table_cols = [col.split()[0] for col in self.table_schemas[table_name]]

        query = "SELECT {} FROM {}".format(", ".join(table_cols), table_name)
        if conditions:
            query += " WHERE "
            conditions_list = []
            # params = []
            for key, value in conditions.items():
                if isinstance(value, dict):
                    # Handle conditions for floats and timestamp
                    for op, val in value.items():
                        if op == "gt":
                            conditions_list.append(f"{key} > {val}")
                        elif op == "lt":
                            conditions_list.append(f"{key} < {val}")
                        elif op == "gte":
                            conditions_list.append(f"{key} >= {val}")
                        elif op == "lte":
                            conditions_list.append(f"{key} <= {val}")
                        elif op == "range":
                            conditions_list.append(f"{key} BETWEEN \"{val[0]}\" AND \"{val[1]}\"")
                else:
                    conditions_list.append(f"{key} = '{value}'")
                    # params.append(value)
            query += " AND ".join(conditions_list)

        # print(query)
        # Execute the query
        if conditions:
            self.cursor.execute(query)
        else:
            self.cursor.execute(query)

        data = []
        for row in self.cursor.fetchall():
            data.append({k: v for k, v in zip(table_cols, row)})

        return data

    def remove_rows_from_table(self, table_name: str, conditions: dict):
        select_query = "delete from {0} where ".format(table_name)
        for item in conditions:
            select_query = select_query + "{0} < '{1}'".format(item, conditions[item])
        self.cursor.execute(select_query)

    def delete_table_data(self,table_name,match_vals):
        table_cols = [col.split()[0] for col in self.table_schemas[table_name]]
        delete_query = "delete from {0} where ".format(table_name)
        delete_vals = []
        for col,value in match_vals.items():
            delete_vals.append(f"{col}" + "=" + f"'{value}'")

        delete_query += " and ".join(delete_vals)

        delete_query += ";"
        # print(delete_query)
        self.cursor.execute(delete_query)

    def delete_full_table_data(self,table_name):
        table_cols = [col.split()[0] for col in self.table_schemas[table_name]]
        delete_query = "delete from {0};".format(table_name)
        # print(delete_query)
        self.cursor.execute(delete_query)

    def populate_tables(self):
        with open(os.path.join(project_dir, "database/db_content.json"), "r") as f:
            table_rows = json.load(f)

        for table in table_rows:
            query = "select count(*) from " + table + ";"
            self.cursor.execute(query)
            table_row_count = self.cursor.fetchone()[0]

            if table_row_count == 0:
                for row in table_rows[table]:
                    self.add_record(table, row)

    def _clear_table(self, table_name):
        delete_string = "delete from " + table_name + ";"
        self.cursor.execute(delete_string)
        self.connection.commit()

    def refresh_records(self, table_name, records):
        if len(records) > 0:
            self._clear_table(table_name)
            for record in records:
                self.add_record(table_name, record)

    def get_table_rows_count(self, table_name, cond=None):
        select_string = "select count(*) from " + table_name

        if cond:
            match_vals = []

            for col in cond:
                match_vals.append(col + "=" + "'" + str(cond[col]) + "'")

            select_string += " where " + " and ".join(match_vals)

        select_string += ";"

        self.cursor.execute(select_string)

        return self.cursor.fetchone()[0]
