import io
import os

import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


def read_excel_from_drive(file_id, sheet_number=0):
    """
    Downloads an Excel file from Google Drive and returns it as a pandas DataFrame.

    Args:
        file_id (str): The Google Drive file ID of the Excel file.
        sheet_number (int): The number of the Sheet, indexed at 0.

    Returns:
        pd.DataFrame: The Excel file content as a pandas DataFrame.
    """
    # Define the scope to access Google Drive
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

    # Path to your service account credentials
    credentials_path = os.getenv("SERVICE_ACCOUNT_CREDENTIALS_PATH")

    # Check if credentials are set
    if credentials_path is None:
        raise ValueError("Credentials path is not set. Please check your .env file.")

    # Authenticate with Google Drive API
    credentials = Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    drive_service = build("drive", "v3", credentials=credentials)

    # Request the file from Google Drive
    request = drive_service.files().get_media(fileId=file_id)

    # Use io.BytesIO to download file content
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")

    # Reset the file pointer to the beginning
    fh.seek(0)

    # Load the Excel file into a pandas dataframe
    df = pd.read_excel(fh, engine="openpyxl", sheet_name=sheet_number)

    return df


def select_gpas_and_binary_usage(df):
    """
    Selects relevant columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame to select columns from.

    Returns:
        pd.DataFrame: DataFrame with selected relevant fields.
    """
    return df.loc[:, ["id", "student_id", "course_code", "course_gpa", "did_use_spark"]]


def select_psm_fields(df):
    """
    Selects relevant fields for PSM analysis: 'id', 'age_at_entry', 'gender', 'ethnicity',
    'course_code', 'course_gpa', and 'did_use_spark'.

    Parameters:
        df (pd.DataFrame): Input DataFrame to select fields from.

    Returns:
        pd.DataFrame: DataFrame with the selected fields.
    """

    # Filter for relevant fields
    return df.loc[
        :,
        [
            "id",
            "age_at_entry",
            "gender",
            "ethnicity",
            "course_code",
            "course_gpa",
            "did_use_spark",
        ],
    ]
