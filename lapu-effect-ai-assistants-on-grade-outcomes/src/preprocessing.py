import janitor


def clean_data(df):
    """
    Cleans and organizes the DataFrame by applying several transformations:
    - Converts columns to snake_case.
    - Renames specific columns for clarity.
    - Strips whitespace from the 'course_code' field.
    - Creates a unique 'id' column based on the DataFrame's index.
    - Moves the 'id' column to the front.
    - Filters out NA fields in the range from 'id' to 'question_49'.

    Parameters:
        df (pd.DataFrame): Input DataFrame to clean and organize.

    Returns:
        pd.DataFrame: Cleaned and organized DataFrame.
    """

    # Convert columns to snake_case
    df = janitor.clean_names(df=df, case_type="snake")

    # Rename id field to student_id (since it is not a unique row ID)
    # Rename other fields for clarity
    df = df.rename(
        columns={
            "id": "student_id",
            "ca": "in_state_or_out_of_state",
            "grade_cde": "grade_code",
            "crs_cde": "course_code",
        }
    )

    # Strip whitespace from course_code field
    df["course_code"] = df["course_code"].str.strip()

    # Create unique ID value
    df = df.assign(id=df.index)

    # Move id column to front
    df.insert(0, "id", df.pop("id"))

    # Filter out NA fields
    df = df.loc[:, "id":"question_49"]

    return df


def create_did_use_spark_field(df):
    """
    Creates a binary 'did_use_spark' field based on 'count_messages' column.

    Parameters:
        df (pd.DataFrame): Input DataFrame to modify.

    Returns:
        pd.DataFrame: DataFrame with 'did_use_spark' field added.
    """

    # Create binary did_use_spark field where count_messages > 2
    df = df.assign(did_use_spark=df["count_messages"] > 2)

    return df


def create_treatment_and_control_groups(
    df, treatment_col="did_use_spark", target_col="course_gpa"
):
    """
    Creates control and treatment groups based on a binary treatment column.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the treatment and target columns.
        treatment_col (str): The column name representing the treatment condition (default: 'did_use_spark').
        target_col (str): The column name representing the target variable (default: 'course_gpa').

    Returns:
        tuple: A tuple containing two numpy arrays: the treatment group and the control group.
    """

    # Create control and treatment groups
    group_treatment = df[df[treatment_col]][target_col].values
    group_control = df[~df[treatment_col]][target_col].values

    return group_treatment, group_control


def drop_missing_genders(df):
    """
    Filters out rows with missing 'gender' values and prints a warning if any are found.

    Parameters:
        df (pd.DataFrame): Input DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame with non-missing 'gender' values.
    """

    count_missing_gender = df["gender"].isna().sum()
    if count_missing_gender > 0:
        print(
            f"***** Warning: {count_missing_gender} rows are missing 'gender' values. *****"
        )

    return df.loc[df["gender"].notna()].reset_index(drop=True)


def drop_missing_gpas(df):
    """
    Filters out rows with missing 'course_gpa' values and prints a warning if any are found.

    Parameters:
        df (pd.DataFrame): Input DataFrame to filter.

    Returns:
        pd.DataFrame: Filtered DataFrame with non-missing 'course_gpa' values.
    """

    count_missing_gpas = df["course_gpa"].isna().sum()
    if count_missing_gpas > 0:
        print(
            f"***** Warning: {count_missing_gpas} rows are missing 'course_gpa' values. *****"
        )

    return df.loc[df["course_gpa"].notna()].reset_index(drop=True)
