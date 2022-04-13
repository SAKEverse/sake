import io, base64
import pandas as pd

# define path to load table
user_table_path = 'example_data/default_table_data.csv'

# load table
original_user_data = pd.read_csv(user_table_path)

# get original columns
user_data_columns = original_user_data.columns

def upload_csv(upload_contents:str):
    """
    Parse string data from dcc upload (csv file)
    and construct pandas dataframe

    PARAMETERS
    ---------
    upload_contents: str, contents from dcc upload object
    
    OUTPUT
    ----------
    df: pd.DataFrame, with user data

    """

    # parse user csv as df
    _, content_string = upload_contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # if columns do not match load original user data
    if set(df.columns) != set(user_data_columns):
        df = original_user_data

    return df