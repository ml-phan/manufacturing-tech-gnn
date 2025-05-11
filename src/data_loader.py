import os
import pandas as pd
import psycopg2

from pathlib import Path
from dotenv import load_dotenv

env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(env_file)

ana_db_params = {
    "host": os.getenv("ANA_DB_HOST"),
    "database": os.getenv("ANA_DB_NAME"),
    "user": os.getenv("ANA_DB_USER"),
    "password": os.getenv("ANA_DB_PASSWORD"),
}
def download_data():
    conn = None
    cur = None
    data = None
    try:
        conn = psycopg2.connect(**ana_db_params)
        cur = conn.cursor()

        # Execute the query
        query = f"""
            SELECT
                m.item_id,
                m.depth,
                m.width,
                m.height,
                m.area,
                m.volume,
                m.bbox_depth,
                m.bbox_height,
                m.bbox_width,
                m.bbox_area,
                m.bbox_volume,
                p.quantity,
                p.download_file_url
            FROM ds_temp.metadata m
            JOIN ds_temp.positions p
                ON m.item_id = p.item_id
            WHERE
                m.item_id IS NOT NULL AND
                m.depth IS NOT NULL AND
                m.width IS NOT NULL AND
                m.height IS NOT NULL AND
                m.area IS NOT NULL AND
                m.volume IS NOT NULL AND
                m.bbox_depth IS NOT NULL AND
                m.bbox_height IS NOT NULL AND
                m.bbox_width IS NOT NULL AND
                m.bbox_area IS NOT NULL AND
                m.bbox_volume IS NOT NULL AND
                p.download_file_url IS NOT null
            ;
        """
        data = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error retrieving data: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
    return data


def save_data(data):
    try:
        data.to_csv("item_data.csv", index=False)
        print("Data saved to item_data.csv")
    except Exception as e:
        print(f"Error saving data: {e}")


def data_processing(data):
    data = data.dropna().reset_index(drop=True)
    data_filter = data[
        data['download_file_url'].str.split("?").str[0].str.contains(
            "step|stp", case=False)]
    data_filter.to_csv("./data/item_data.csv", index=False)


