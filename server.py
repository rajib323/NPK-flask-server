from flask import Flask,request,json,jsonify,Response
import openmeteo_requests
import requests_cache
import pandas as pd
from sklearn.linear_model import Ridge
from retry_requests import retry
import pickle
from datetime import date


app = Flask(__name__)
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)
url = "https://archive-api.open-meteo.com/v1/archive"

hybrid_model = pickle.load(open('crop_hybrid.pkl' , 'rb'))
logreg_model = pickle.load(open('crop_logreg.pkl' , 'rb'))
naive_model = pickle.load(open('crop_naive.pkl' , 'rb'))
rfc_model = pickle.load(open('crop_rfc.pkl' , 'rb'))


def backtest(weather, model, predictors,target_name, start=7300, step=90):
    all_predictions = []
    
    for i in range (start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test  = weather.iloc[i:(i+step),:]
        
        model.fit(train[predictors], train[target_name])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test[target_name], preds], axis=1)
        combined.columns = ["actual","prediction"]
        combined["diff"]=(combined["prediction"] - combined["actual"]).abs()
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)

def getWeatherData(url,params):
    response = openmeteo.weather_api(url, params=params)[0]

    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(2).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(3).ValuesAsNumpy()

    daily_data = {"time": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["tmax"] = daily_temperature_2m_max
    daily_data["tmin"] = daily_temperature_2m_min
    daily_data["tmean"] = daily_temperature_2m_mean
    daily_data["rain"] = daily_rain_sum

    weather = pd.DataFrame(data = daily_data)
    weather.set_index('time',inplace=True)  
    return weather

def compute_rolling(weather, horizon, col): 
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label],weather[col])
    return weather

def pct_diff(old, new):
    return (new - old) / old

def expand_mean(df):
    return df.expanding(1).mean()

def predictTemp(weather):
    shift_value = -10

    null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]
    valid_columns = weather.columns[null_pct<0.05]
    weather = weather[valid_columns].copy()
    weather.columns = weather.columns.str.lower()
    weather = weather.ffill()
    weather.index = pd.to_datetime(weather.index)

    weather["target_tmax"]=weather.shift(shift_value)["tmax"]
    weather["target_tmin"]=weather.shift(shift_value)["tmin"]
    weather["target_tmean"]=weather.shift(shift_value)["tmean"]
    weather["target_rain"]=weather.shift(shift_value)["rain"]

    weather=weather.ffill()

    rr = Ridge(alpha = .1)

    predictors = weather.columns[~weather.columns.isin(["target_tmax","target_tmin","target_tmean","target_rain"])]

    predict_tmax = backtest(weather, rr, predictors,"target_tmax")
    predict_tmin = backtest(weather,rr,predictors,"target_tmin")
    predict_tmean = backtest(weather,rr,predictors,"target_tmean")
    predict_rain = backtest(weather,rr,predictors,"target_rain")

    rolling_horizons = [3,14]

    for horizon in rolling_horizons:
        for col in ["tmax","tmin","tmean","rain"]:
            weather = compute_rolling(weather,horizon,col)

    weather = weather.iloc[14:,:]
    weather= weather.fillna(0)

    for col in ["tmax","tmin","tmean","rain"]:
        weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month,group_keys=False).apply(expand_mean)
        weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year,group_keys=False).apply(expand_mean)

    predictors = weather.columns[~weather.columns.isin(["target_tmax","target_tmin","target_tmean","target_rain"])]

    predict_tmax = backtest(weather,rr,predictors,"target_tmax")
    predict_tmin = backtest(weather,rr,predictors,"target_tmin")
    predict_tmean = backtest(weather,rr,predictors,"target_tmean")
    predict_rain = backtest(weather,rr,predictors,"target_rain")

    predict_tmax = predict_tmax['prediction'].iloc[-11:-1].copy()
    predict_tmin = predict_tmin['prediction'].iloc[-11:-1].copy()
    predict_tmean = predict_tmean['prediction'].iloc[-11:-1].copy()
    predict_rain = predict_rain['prediction'].iloc[-11:-1].copy()

    predict_tmax.index += pd.Timedelta(days=-shift_value)
    predict_tmin.index += pd.Timedelta(days=-shift_value)
    predict_tmean.index += pd.Timedelta(days=-shift_value)
    predict_rain.index += pd.Timedelta(days=-shift_value)
    
    
    df=pd.merge(left=predict_tmax,right=predict_tmin,how="inner",on=["time"])
    df.rename(columns={"prediction_x":"tmax","prediction_y":"tmin"},inplace=True)
    new_df=pd.merge(left=df,right=predict_tmean,how="inner",on=["time"])
    new_df.rename(columns={"prediction":"tmean"},inplace=True)
    
    return new_df
    
@app.route("/predict-weather",methods=['POST'])
def predictWeather():
    data=request.get_data()
    reqBody=json.loads(data)
    print(date.today())
    
    params = {
        "latitude": reqBody["latitude"],
        "longitude": reqBody["longitude"],
        "start_date": "2000-01-01",
        "end_date": str(date.today()),
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "rain_sum"],
        "timezone": "auto"
    }
    print("Fetching For : ",params)
    weather=getWeatherData(url,params)
    prediction=predictTemp(weather)
    
    
    predictData=prediction.to_json(orient='table',date_format='iso')
    
    # .to_json(orient='table',date_format='iso')
    return predictData


@app.route("/predict-crop",methods=["POST"])
def predictCrop():
    reqBody=request.get_data()
    data=json.loads(reqBody)["data"]
    
    naive_prob=naive_model.predict_proba([data])
    rf_prob = rfc_model.predict_proba([data])
    log_prob = logreg_model.predict_proba([data])
    avg_prob = (naive_prob + rf_prob+log_prob) / 3
    classes = hybrid_model.classes_
    probabilities = avg_prob[0]  
    class_probabilities = list(zip(classes, probabilities))
    class_probabilities.sort(key=lambda x: x[1], reverse=True)

    top_n = 6
    top_classes = [crop_name[0] for crop_name in class_probabilities[:top_n]]
    
    top_probabilities = [crop_name[1] for crop_name in class_probabilities[:top_n]]

    predictionData=[]
    for crop_name, prob in zip(top_classes, top_probabilities):
        predResult={
            "crop_name":crop_name,
            "probability":prob*100
        }
        predictionData.append(predResult);        
    
    
    return jsonify(
        {
            "data":predictionData   
        }
    )


if __name__ == '__main__':
    app.run()