import pandas as pd

TRAIN_DATA_PATH = "data/raw/KDDTrain+.txt"

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack",
    "difficulty"
]


def load_training_data():
    """
    Load the NSL-KDD training dataset.

    Returns:
        pandas.DataFrame
    """
    df = pd.read_csv(
        TRAIN_DATA_PATH,
        names=COLUMN_NAMES
    )

    return df

def inspect_data(df):
    """
    Display important information about the dataset.
    """

    print("\n" + "=" * 70)
    print("DATASET SHAPE")
    print("=" * 70)
    print(df.shape)

    print("\n" + "=" * 70)
    print("FIRST FIVE ROWS")
    print("=" * 70)
    print(df.head())

    print("\n" + "=" * 70)
    print("LAST FIVE ROWS")
    print("=" * 70)
    print(df.tail())

    print("\n" + "=" * 70)
    print("DATA TYPES")
    print("=" * 70)
    print(df.dtypes)

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    print(df.isnull().sum())

    print("\n" + "=" * 70)
    print("DUPLICATE ROWS")
    print("=" * 70)

    print(df.duplicated().sum())

    print("\n" + "=" * 70)
    print("ATTACK DISTRIBUTION")
    print("=" * 70)

    print(df["attack"].value_counts())

    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    print(df.describe())

def main():

    df = load_training_data()

    print("\nDataset Loaded Successfully!\n")

    inspect_data(df)


if __name__ == "__main__":
    main()