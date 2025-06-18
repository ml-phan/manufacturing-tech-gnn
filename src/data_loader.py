import joblib
import numpy as np
import os
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from jedi.settings import fast_parser
from sqlalchemy import create_engine, Engine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


env_file = find_dotenv()
load_dotenv(dotenv_path=env_file)

ana_db_params = {
    "dbname": os.getenv("PG_DB_NAME_ANA"),
    "user": os.getenv("PG_USERNAME_ANA"),
    "host": os.getenv("PG_HOST_ANA"),
    "password": os.getenv("PG_PASSWORD"),
    "port": os.getenv("PG_PORT"),
}

prod_db_params = {
    "dbname": os.getenv("PG_DB_NAME_PROD"),
    "user": os.getenv("PG_USERNAME_PROD"),
    "host": os.getenv("PG_HOST_PROD"),
    "password": os.getenv("PG_PASSWORD"),
    "port": os.getenv("PG_PORT"),
}

ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
SECRET_KEY = os.getenv('AWS_SECRET_KEY')
BUCKET_NAME_PARTS = os.getenv('AWS_BUCKET_NAME_PARTS_PROD')
AWS_BUCKET_NAME_ATTACHMENTS = os.getenv('AWS_BUCKET_NAME_ATTACHMENTS_PROD')
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def create_sqlalchemy_engine(db_params):
    """Create a SQLAlchemy engine for PostgreSQL database connection.
    Args:
        db_params (dict): Database connection parameters.
    Returns:
        sqlalchemy.engine.base.Engine: SQLAlchemy engine object.
    """
    conn_str = (
        f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}"
        f"@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    )
    return create_engine(conn_str)


def fetch_mapping_from_db(
        table_name: str,
        file_name: str,
        db_engine: Engine,
        data_dir: Path = DATA_DIR,
        columns: str = "ref_id, name",
        update: bool = False,
) -> pd.DataFrame:
    """Generic function to fetch mapping from a database table and cache it locally as CSV.
    Args:
        table_name (str): Name of the table to fetch data from.
        file_name (str): Name of the CSV file to save the data.
        db_engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine object.
        data_dir (Path): Directory to save the CSV file.
        columns (str): Columns to select from the table.
        update (bool): Whether to update the CSV file if it already exists.
    Returns:
        pd.DataFrame: DataFrame containing the fetched data.
    """

    data_file = data_dir / file_name
    if data_file.is_file() and not update:
        return pd.read_csv(data_file)

    try:
        query = f"SELECT {columns} FROM public.{table_name};"
        df = pd.read_sql_query(query, db_engine)
        df.to_csv(data_file, index=False)
        return df
    except Exception as e:
        print(f"Error retrieving data from {table_name}: {e}")
        return pd.DataFrame()


def get_technology_mapping(update=False) -> pd.DataFrame:
    df = fetch_mapping_from_db(
        table_name="technology",
        file_name="technology_mapping.csv",
        db_engine=create_sqlalchemy_engine(prod_db_params),
        columns="technology, technology_id",
        update=update,
    )
    df.columns = ["name", "ref_id"]
    return df


def get_material_mapping(update=False) -> pd.DataFrame:
    return fetch_mapping_from_db(
        table_name="material",
        file_name="material_mapping.csv",
        db_engine=create_sqlalchemy_engine(prod_db_params),
        update=update,
    )


def get_post_processing_mapping(update=False) -> pd.DataFrame:
    return fetch_mapping_from_db(
        table_name="finish",
        file_name="post_processing_mapping.csv",
        db_engine=create_sqlalchemy_engine(prod_db_params),
        update=update,
    )


def download_data(update=False) -> pd.DataFrame:
    """Download data from PostgreSQL database and save it to a CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the downloaded data.
    """
    data = None
    data_file = DATA_DIR / "item_data.csv"
    if not data_file.is_file() or update:
        print("Downloading data from PostgreSQL database...")
        try:
            # Create SQLAlchemy engine
            engine = create_sqlalchemy_engine(ana_db_params)
            # Execute the query
            query = f"""
                SELECT
                    m.item_id, m.depth, m.width,m.height, m.area, m.volume,
                    m.bbox_depth, m.bbox_height, m.bbox_width, m.bbox_area,
                    m.bbox_volume,
                    p.quantity, p.technology_id, p.material_id,
                    p.post_processing_id, p.download_file_url
                FROM ds_temp.metadata m
                JOIN ds_temp.positions p
                    ON m.item_id = p.item_id
                WHERE
                    m.item_id IS NOT NULL AND m.depth IS NOT NULL AND
                    m.width IS NOT NULL AND m.height IS NOT NULL AND
                    m.area IS NOT NULL AND m.volume IS NOT NULL AND
                    m.bbox_depth IS NOT NULL AND m.bbox_height IS NOT NULL AND
                    m.bbox_width IS NOT NULL AND m.bbox_area IS NOT NULL AND
                    m.bbox_volume IS NOT NULL AND p.quantity IS NOT NULL AND
                    p.technology_id IS NOT NULL AND 
                    p.material_id IS NOT NULL AND 
                    p.post_processing_id IS NOT NULL AND
                    p.download_file_url IS NOT null
                ;
            """
            data = pd.read_sql_query(query, engine)
            data.to_csv(data_file, index=False)
        except Exception as e:
            print(f"Error retrieving data: {e}")
    else:
        data = pd.read_csv(data_file)
    return data


def data_processing(data: pd.DataFrame) -> pd.DataFrame:
    """Process the downloaded data and save it to a CSV file.
    Args:
        data (pd.DataFrame): DataFrame containing the downloaded data.
    Returns:
        pd.DataFrame: Processed DataFrame.
    """

    # Remove rows with NaN values
    data = data.dropna().reset_index(drop=True)

    # Remove rows containing files that are not step files
    data_filter = data[
        data['download_file_url'].str.split("?").str[0].str.contains(
            "step|stp", case=False)]
    # data_filter.to_csv("./data/item_data.csv", index=False)

    # Add number of faces, edges, and vertices to the dataframe
    file_paths = [
        "./data/step_file_features.csv",
        "./data/step_file_features_2.csv",
        "./data/step_file_features_3.csv",
    ]
    # Read and combine all into one DataFrame
    df_combined = pd.concat([pd.read_csv(file) for file in file_paths],
                            ignore_index=True)
    df_combined = df_combined.dropna().reset_index(drop=True)
    df_combined.item_id = df_combined.item_id.astype(int)

    dataset = pd.merge(df_combined, data_filter, on="item_id", how="inner")

    technology_mapping = get_technology_mapping()
    material_mapping = get_material_mapping()
    post_processing_mapping = get_post_processing_mapping()

    dataset = pd.merge(
        dataset,
        technology_mapping.rename(
            columns={'ref_id': 'technology_id', 'name': 'technology_name'}),
        on='technology_id',
        how='left'
    )

    # Merge mat_map to get material_name
    dataset = pd.merge(
        dataset,
        material_mapping.rename(
            columns={'ref_id': 'material_id', 'name': 'material_name'}),
        on='material_id',
        how='left'
    )

    # Create "Is_CNC" column for binary classification
    dataset["is_cnc"] = dataset["technology_name"].str.contains(
        "CNC",
        case=False,
        na=False
    ).astype(int)

    # Encode the labels only if there is no existing encoder file
    encoder_path = "./models/label_encoder.pkl"

    if os.path.exists(encoder_path):
        le = joblib.load(encoder_path)
        print("Loaded existing LabelEncoder.")
    else:
        le = LabelEncoder()
        le.fit(dataset["technology_name"])
        joblib.dump(le, encoder_path)
        print("Created and saved new LabelEncoder.")

    dataset["multiclass_labels"] = le.transform(dataset["technology_name"])


    dataset_file = DATA_DIR / "dataset.csv"
    dataset.to_csv(dataset_file, index=False)

    return dataset


def remove_minority_class(
        data: pd.DataFrame,
        percentage_threshold: int = 1
) -> pd.DataFrame:
    """Remove minority class from the dataset.
    Args:
        data (pd.DataFrame): DataFrame containing the dataset.
        percentage_threshold (int): Minimum percentage threshold for class removal.
    Returns:
        pd.DataFrame: DataFrame with the minority class removed.
    """
    # Count occurrences of each class
    class_percentages = data['technology_name'].value_counts(normalize=True).sort_index().sort_values(ascending=False) * 100

    # Identify classes to keep
    classes_to_keep = class_percentages[class_percentages >= percentage_threshold].index.tolist()

    # Filter the DataFrame to keep only the selected classes
    data = data[data['technology_name'].isin(classes_to_keep)].reset_index(
        drop=True)

    return data

def get_data():
    """Get the data from the database and process it.
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Download data
    data = download_data(update=False)

    # Process data
    data = data_processing(data)

    # Remove minority class
    data = remove_minority_class(data, percentage_threshold=1)
    return data

