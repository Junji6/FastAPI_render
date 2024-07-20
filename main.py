#必要なライブラリをインポート
from fastapi import FastAPI

#追加：PydenticからBaseModelとインポート
from pydantic import BaseModel 
from sklearn.ensemble import RandomForestClassifier
import pickle

#FastAPIのインスタンス化
app = FastAPI()

#入力するデータ型の定義
class iris(BaseModel):
    sepal_length:float
    sepal_width :float
    petal_length:float
    petal_width :float

#学習モデルの読み込み
model = pickle.load(open('model_iris','rb'))

#ルートディレクトリへのGETでiris_predictionの表示
@app.get('/')
async def index():
    return {"Iris" : "iris_prediction"}

#POSTが送信されたとき（入力）と予測値（出力）の定義
@app.post('/make_predictions')
async def make_predictions(features:iris):
    return({'prediction':str(model.predict([[features.sepal_length,features.sepal_width,features.petal_length,features.petal_width]])[0])})
