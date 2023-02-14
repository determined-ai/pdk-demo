import pandas as pd
import torch

from determined.experimental import Determined

# Apply standard scaling to all numerical columns of df, using mean and std from reference_df
# Make sure not to pass a scaled "reference_df" as argument, since we use its values to scale df
def scale_data(df, reference_df, numerical_cols):
    for col in numerical_cols:
        mean = reference_df[col].mean()
        std = reference_df[col].std()
        df[col] = (df[col] - mean) / std
    
    return df

# One-hot encode all categorical features, assuming we know all values they may take
def encode_categories(df):
    expected_categories = {}
    expected_categories["new_cell"] = ['U','Y','N']
    expected_categories["asl_flag"] = ['N','Y']
    expected_categories["area"] = ['NORTHWEST/ROCKY MOUNTAIN AREA','GREAT LAKES AREA','CHICAGO AREA',
     'NEW ENGLAND AREA','DALLAS AREA','CENTRAL/SOUTH TEXAS AREA',
     'TENNESSEE AREA','MIDWEST AREA','PHILADELPHIA AREA','OHIO AREA',
     'HOUSTON AREA','SOUTHWEST AREA','NEW YORK CITY AREA',
     'ATLANTIC SOUTH AREA','SOUTH FLORIDA AREA','CALIFORNIA NORTH AREA',
     'DC/MARYLAND/VIRGINIA AREA','NORTH FLORIDA AREA','LOS ANGELES AREA']
    expected_categories["dualband"] = ['Y','N','T']
    expected_categories["refurb_new"] = ['N','R']
    expected_categories["hnd_webcap"] = ['WCMB','UNKW','WC']
    expected_categories["marital"] = ['S','M','A','U','B']
    expected_categories["ethnic"] = ['N','U','I','S','F','J','Z','M','H','G','D','O','R','B','P','X','C']
    expected_categories["kid0_2"] = ['U','Y']
    expected_categories["kid3_5"] = ['U','Y']
    expected_categories["kid6_10"] = ['U','Y']
    expected_categories["kid11_15"] = ['U','Y']
    expected_categories["kid16_17"] = ['U','Y']
    expected_categories["creditcd"] = ['Y','N']
    
    for col in expected_categories:
        categorical_col = pd.Categorical(df[col], categories=expected_categories[col], ordered=False)
        one_hot_cols = pd.get_dummies(categorical_col, prefix=col)
        df.drop(col, axis=1, inplace=True)
        df = pd.concat([df, one_hot_cols], axis=1)
    
    return df

# Make sure not to pass a scaled "reference_df" as argument, since we use its values to scale df
def preprocess_dataframe(df, reference_df, numerical_cols):
    df = scale_data(df, reference_df, numerical_cols)
    df = encode_categories(df)
    return df

def predict_and_evaluate(model, df_to_predict, data_files=["data/cleaned_data.csv"]):
    
    # Read csv files one by one and concatenate them to get the full dataset
    reference_df = pd.read_csv(files[0])
    
    for file in data_files[1:]:
        partial_df = pd.read_csv(file)
        reference_df = pd.concat([reference_df, partial_df], axis=0)
        reference_df.reset_index(drop=True, inplace=True)
    
    df = df_to_predict.copy()
    
    object_cols = list(reference_df.columns[reference_df.dtypes.values == "object"])
    int_cols = list(reference_df.columns[reference_df.dtypes.values == "int"])
    float_cols = list(reference_df.columns[reference_df.dtypes.values == "float"])

    # Churn will be the label, no need to preprocess it
    int_cols.remove("churn")

    numerical_cols = int_cols+float_cols
    
    df = preprocess_dataframe(df, reference_df, numerical_cols)
    feature_cols = list(df.columns)
    label_col = "churn"
    feature_cols.remove(label_col)
    
    input_tensor = torch.Tensor(df[feature_cols].values)
    label_tensor = torch.Tensor(df[[label_col]].values)
    
    model.eval()
    with torch.no_grad():
        preds = model(input_tensor)
    
    preds[preds < 0.5] = 0.0
    preds[preds >= 0.5] = 1.0
    accuracy = float(torch.sum(label_tensor == preds) / len(preds))
    
    return preds.numpy(), accuracy


def check_model(model_name):

    if len(Determined().get_models(name=model_name)) > 0:
        model = Determined().get_models(name=model_name)[0]
    else:
        model = Determined().create_model(model_name)
        
    return model