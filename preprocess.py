import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def readdata(divipath, pranavpath, finalpath):
    '''
    Reads data and preprocesses it for later use. This function needs to be called only once in the entire process
    as this creates a final dataset as a csv file which can be accessed directly later on.

    1) User ID is assigned.
    2) The dataset of both the users is merged into one.
    3) Returns final dataset
    '''
    divi = pd.read_csv(divipath)
    pranav = pd.read_csv(pranavpath)

    divi.to_csv("Data/DiviFinal.csv", index=False)
    pranav.to_csv("Data/PranavFinal.csv", index=False)

    pranav["User"] = 1
    divi["User"] = 0

    final = pd.concat([divi, pranav], ignore_index=True)
    
    final.to_csv(finalpath, index=False)

    return final


# def normalize(finalpath):
#     '''
#     1) Uses a MinMaxScaler to normalize the press_time and release_time values
#     2) These values were stored in a TimeStamp format and needed to converted into numerical values.
#     '''
#     data = pd.read_csv(finalpath)

#     scaler = MinMaxScaler()
#     data[["press_time","release_time", "dd_time", "flight_time","hold_time"]] = scaler.fit_transform(data[["press_time","release_time", "dd_time", "flight_time","hold_time"]])
    
#     return data


def create_sequences(data):
    '''
    This function creates multiple sequences of 10 key logs in order to feed them to the LSTM model.
    Each sequence contains 10 consecutive keystrokes from the same user and is labeled accordingly.
    '''
    X = []  # key sequences
    y = []  # corresponding labels
    
    feature_cols = [col for col in data.columns if col != 'User']
    users = data["User"].unique()
    
    for user in users:
        user_data = data[data["User"] == user].copy()
        totalkeys = len(user_data)

        remainder = totalkeys % 10
        if remainder != 0:
            user_data = user_data.iloc[:-remainder]
        
        for i in range(0, len(user_data) - 9, 10):
            sequence = user_data.iloc[i:i+10][feature_cols].values  # shape: (10, num_features)
            X.append(sequence)
            y.append(user)  # or encode as 0/1 if needed

    return np.array(X), np.array(y)


# def one_hot_encode_keys_raw_data(data):
#     """
#     One-hot encodes the 'key' column in a raw DataFrame and returns the updated DataFrame.

#     Args:
#         data (pd.DataFrame): The raw keystroke DataFrame with a 'key' column.

#     Returns:
#         pd.DataFrame: Updated DataFrame with key one-hot encoded.
#         OneHotEncoder: Fitted encoder (for inverse transform or future use).
#     """
#     onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     key_encoded = onehot_encoder.fit_transform(data[['key']])

#     # Create column names for one-hot encoded keys
#     key_columns = onehot_encoder.get_feature_names_out(['key'])
#     key_df = pd.DataFrame(key_encoded, columns=key_columns)

#     # Concatenate one-hot key columns with other features
#     data_encoded = pd.concat([key_df, data.drop(columns=['key'])], axis=1)
    
#     return data_encoded, onehot_encoder

def split_normalize(data):

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data.drop("User", axis = 1)
    y = data["User"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    pd.DataFrame(X_train).to_csv("Data/XTrain.csv")
    pd.DataFrame(y_train).to_csv("Data/YTrain.csv")

    pd.DataFrame(X_test).to_csv("Data/XTest.csv")
    pd.DataFrame(y_test).to_csv("Data/YTest.csv")
    
    return X_train, X_test, y_train, y_test

def preprocess_and_create_sequences(divipath, pranavpath, finalpath, trainpath=None, testpath=None):
    data = readdata(divipath, pranavpath, finalpath)

    data = data.drop(columns=['key'])

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["User"]
    )

    scaler = StandardScaler()
    train_data.iloc[:, 1:-1] = scaler.fit_transform(train_data.iloc[:, 1:-1])
    test_data.iloc[:, 1:-1] = scaler.transform(test_data.iloc[:, 1:-1])
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(train_data)
    X_test_seq, y_test_seq = create_sequences(test_data)

    # Label encode
    label_encoder = LabelEncoder()
    y_train_seq_encoded = label_encoder.fit_transform(y_train_seq)
    y_test_seq_encoded = label_encoder.transform(y_test_seq)

    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, label_encoder

if __name__ == "__main__":
    divi_path = "Data/DiviFinal.csv"
    pranav_path = "Data/PranavFinal.csv"
    final_path = "Data/Final.csv"
    X_train, X_test, y_train, y_test, label_encoder = preprocess_and_create_sequences(divi_path, pranav_path, final_path)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Label classes: {label_encoder.classes_}")


