from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

app = FastAPI(title="Housing Sale price Prediction API")

@app.get("/")
def home():
    return HTMLResponse("""
    <html>
        <body>
            <h2>Upload CSV para predição de preço de venda</h2>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Enviar">
            </form>
        </body>
    </html>
    """)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(BytesIO(contents), sep=None, engine='python')
    except Exception:
        df = pd.read_csv(BytesIO(contents))
    df.columns = df.columns.str.strip()
    predictions = predict_from_df(df, "best_model.pkl")
    result = []
    for i, row in df.iterrows():
        item = {
            "PID": row.get("PID"),
            "SalePrice": float(predictions[i])
        }
        result.append(item)
    return {"predictions": result}

def predict_from_df(df, model_path):
    X_train_columns = [
        'Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area',
        'Total Bsmt SF', '1st Flr SF', 'Year Built', 'Full Bath',
        'Year Remod/Add', 'Mas Vnr Area', 'Neighborhood_Blueste',
        'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr',
        'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards',
        'Neighborhood_Gilbert', 'Neighborhood_Greens', 'Neighborhood_GrnHill',
        'Neighborhood_IDOTRR', 'Neighborhood_Landmrk', 'Neighborhood_MeadowV',
        'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill',
        'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt',
        'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer',
        'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr',
        'Neighborhood_Timber', 'Neighborhood_Veenker', 'Exter Qual_Fa',
        'Exter Qual_Gd', 'Exter Qual_TA', 'Kitchen Qual_Fa', 'Kitchen Qual_Gd',
        'Kitchen Qual_TA', 'Bsmt Qual_Fa', 'Bsmt Qual_Gd', 'Bsmt Qual_None',
        'Bsmt Qual_Po', 'Bsmt Qual_TA', 'Garage Finish_None',
        'Garage Finish_RFn', 'Garage Finish_Unf', 'Garage Qual_Fa',
        'Garage Qual_Gd', 'Garage Qual_None', 'Garage Qual_Po',
        'Garage Qual_TA', 'House Style_1.5Unf', 'House Style_1Story',
        'House Style_2.5Fin', 'House Style_2.5Unf', 'House Style_2Story',
        'House Style_SFoyer', 'House Style_SLvl', 'Sale Condition_AdjLand',
        'Sale Condition_Alloca', 'Sale Condition_Family',
        'Sale Condition_Normal', 'Sale Condition_Partial', 'MS Zoning_C (all)',
        'MS Zoning_FV', 'MS Zoning_I (all)', 'MS Zoning_RH', 'MS Zoning_RL',
        'MS Zoning_RM'
    ]

    categoric_cols_none = [
        'Alley', 'Mas Vnr Type', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
        'BsmtFin SF 1', 'BsmtFin Type 2', 'Fireplace Qu', 'Garage Type',
        'Garage Finish', 'Garage Qual', 'Garage Cond', 'Pool QC',
        'Fence', 'Misc Feature'
    ]

    numeric_cols_zero = [
        'Mas Vnr Area', 'BsmtFin Type 1', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF',
        'Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath',
        'Garage Cars', 'Garage Area', 'Garage Yr Blt'
    ]

    categorical_columns_to_encode = [
        'Neighborhood', 'Exter Qual', 'Kitchen Qual', 'Bsmt Qual',
        'Garage Finish', 'Garage Qual', 'House Style', 'Sale Condition', 'MS Zoning'
    ]

    def encode_categorical(df, categorical_columns_to_encode):
        df_encoded = df.copy()
        for col in categorical_columns_to_encode:
            if col in df_encoded and df_encoded[col].dtype == 'bool':
                df_encoded[col] = df_encoded[col].astype(int)
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=[col for col in categorical_columns_to_encode if col in df_encoded],
            drop_first=True,
            dtype=int
        )
        return df_encoded

    for col in categoric_cols_none:
        if col in df:
            df[col] = df[col].fillna("None")

    for col in numeric_cols_zero:
        if col in df:
            df[col] = df[col].fillna(0)

    if "Lot Frontage" in df and "Neighborhood" in df:
        df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(
            lambda x: x.fillna(x.median())
        )
        df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())

    df = encode_categorical(df, categorical_columns_to_encode)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in X_train_columns:
        if col not in df:
            df[col] = 0

    df = df[X_train_columns]

    best_model = joblib.load(model_path)
    y_pred_log = best_model.predict(df)
    y_pred = np.expm1(y_pred_log)
    return y_pred
