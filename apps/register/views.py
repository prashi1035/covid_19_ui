import os
from django.shortcuts import render, HttpResponse, redirect
from django.contrib import messages
from django.contrib.auth import logout
#import bcrypt
from .models import User

import pandas as pd
import numpy as np
import kaggle
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from pmdarima import auto_arima
import pickle
import datetime
#FILE_PATH = r"\\chnrmz-na-v504.asia-pac.shell.com\QTHOME002\Prashanthi.Bojanapu\Home\prashanthi\pdfextraction\COE\AI%20Engineering\data_trekkers\Input_data\covid_19_data.csv"
#kaggle.api.authenticate()

dataset = 'sudalairajkumar/novel-corona-virus-2019-dataset'
cwd = os.getcwd()
path_to_save = cwd


def extract_data():
    '''This function access kaggle site and down load the sudalairajkumar/novel-corona-virus-2019-dataset
    data set in the current working directory and returns the covid_19_data.csv dataframe'''
    
    response = ""
    try:
        print("Fetching data from kaggle - This might take sometime")
        data = kaggle.api.dataset_download_files(dataset,path=path_to_save,unzip = True)
        df_covid19 = pd.read_csv(os.path.join(cwd, 'covid_19_data.csv'), encoding='ISO-8859-1')
        response = df_covid19
    except Exception as error:
        response = "Failed"
    finally:
        print("Fetched data successfully")
        return response
    
    
covid_kaggle_data = pd.read_csv(r'./covid_19_data.csv')


def index(request):
    return render(request, 'register/index.html')


def register(request):
    errors = User.objects.validator(request.POST)
    if len(errors):
        for tag, error in errors.iteritems():
            messages.error(request, error, extra_tags=tag)
        return redirect('/')

    # hashed_password = bcrypt.hashpw(request.POST['password'].encode('utf-8'), bcrypt.gensalt())
    hashed_password = request.POST['password']
    user = User.objects.create(first_name=request.POST['first_name'], last_name=request.POST['last_name'], password=hashed_password, email=request.POST['email'])
    user.save()
    request.session['id'] = user.id
    print("*"*20,user)
    return redirect('/success')


def login(request):
    if (User.objects.filter(email=request.POST['login_email']).exists()):
        user = User.objects.filter(email=request.POST['login_email'])[0]
        # if (bcrypt.checkpw(request.POST['login_password'].encode('utf-8'), user.password.encode('utf-8'))):
        if request.POST['login_password'] ==  user.password:
            request.session['id'] = user.id
            return redirect('/covid')
    return redirect('/')


def homepage(request):
    return redirect('/covid')


def logout_req(request):
    logout(request)
    return redirect('/')


def success(request):
    user = User.objects.get(id=request.session['id'])
    context = {
        "user": user
    }
    return render(request, 'register/success.html', context)


def covid(request):
    user = User.objects.get(id=request.session['id'])
    dataframe = covid_kaggle_data
    countries = dataframe['Country/Region'].unique()
    countries.sort()
    context = {
        "user": user,
        'countries' : countries
    }
    return render(request, 'register/covid.html', context)


def covid_predictions(request):
    ''' This function will give the predictions for the selected country'''
    historical = False
    if "historical" not in request.POST:
        historical = False
    if "historical" in request.POST:
        historical = request.POST["historical"]
        if historical == "True":
            historical = True 
    user = User.objects.get(id=request.session['id'])
    df_pred, hist_df = predict_covid_cases(request.POST["country"], is_history=historical)

    context = {
        "user": user,
        "country": request.POST["country"],
        "predictions": df_pred.to_html(),
        "historical": hist_df.to_html()
    }
    return render(request, 'register/success_pred.html', context)
    

def get_data(country):
    """
    This is a method to get input data
    :arg:
      country - name of country, if none returns all the data
    :return: pandas dataframe of covid 19
    """
    # data = kaggle_util.extract_data()
    data = covid_kaggle_data
    data_agg = data.groupby(['ObservationDate', 'Country/Region']).agg({'Confirmed': 'sum'}).reset_index()
    data_agg.columns = ['date', 'country', 'no_cases']
    data_agg['date'] = pd.to_datetime(data_agg['date'])
    country_df = data_agg[data_agg['country'] == country].sort_values('date')
    country_df.index = country_df['date']
    return country_df[['date', 'no_cases']]


def get_arima_model(series):
    print("Creating ARIMA model for prediction")
    model = auto_arima(series, trace=False, error_action='ignore', suppress_warnings=True)
    model.fit(series)
    print("Model creation successful: " + str(model))
    return model


def get_forecast(model, no_days):
    return model.predict(n_periods=no_days).astype(np.int)


def get_rolling_window_dates(dates_list, min_period, forecast_period):
    rolling_window_list = []
    list_exhaust = True
    last_index = 0
    while list_exhaust:
        if len(rolling_window_list) == 0:
            window_dict = dict()
            window_dict['train_from'] = dates_list[0]
            window_dict['train_to'] = dates_list[min_period-1]
            window_dict['test_from'] = dates_list[min_period]
            window_dict['test_to'] = dates_list[min_period + forecast_period -1]
            last_index = min_period + forecast_period -1
            rolling_window_list.append(window_dict.copy())
        else:
            window_dict = dict()
            window_dict['train_from'] = dates_list[0]
            window_dict['train_to'] = dates_list[last_index]
            window_dict['test_from'] = dates_list[last_index + 1]
            if (last_index + forecast_period - 1) < len(dates_list)-1:
                window_dict['test_to'] = dates_list[last_index + forecast_period - 1]
                last_index = last_index + forecast_period - 1
                rolling_window_list.append(window_dict.copy())
            else:
                window_dict['test_to'] = dates_list[-1]
                rolling_window_list.append(window_dict.copy())
                break
    return rolling_window_list


def slice_df_by_time(df, start_date, end_date):
    return df[(df['date'] >= pd.to_datetime(start_date)) &
              (df['date'] <= pd.to_datetime(end_date))]


def get_historical_performance(df, min_period=70, forecast_period=7):
    print("\n######## Historical Performance - Rolling window prediction ###########\n")
    hist_perf_list = []
    rolling_windows = get_rolling_window_dates(list(df['date'].values), min_period, forecast_period)
    for this_window in rolling_windows:
        train_data = slice_df_by_time(df, this_window['train_from'], this_window['train_to'])
        test_data = slice_df_by_time(df, this_window['test_from'], this_window['test_to'])
        this_run_model = get_arima_model(train_data['no_cases'])
        this_run_forecast = get_forecast(this_run_model, len(test_data))
        r_sq = r2_score(test_data['no_cases'].values, this_run_forecast)
        rms = sqrt(mean_squared_error(test_data['no_cases'].values, this_run_forecast))
        if len(test_data == forecast_period):
            perf_dict = dict()
            perf_dict['train_period'] = str(train_data['date'][0].date()) + " to " + str(train_data['date'][-1].date())
            perf_dict['test_period'] = str(test_data['date'][0].date()) + " to " + str(test_data['date'][-1].date())
            perf_dict['actual_cases'] = test_data['no_cases'][-1]
            perf_dict['predicted_cases'] = this_run_forecast[-1]
            perf_dict['R square'] = r_sq
            perf_dict['RMSE'] = rms
            print("Train from : " + str(train_data['date'][0].date()) + " to " + str(train_data['date'][-1].date()))
            print("Test from : " + str(test_data['date'][0].date()) + " to " + str(test_data['date'][-1].date()))
            print("Actual cases (end of period) : " + str(test_data['no_cases'][-1]))
            print("Predicted cases (end of period) : " + str(this_run_forecast[-1]))
            print("R-Squared (for daily level prediction) : " + str(r_sq))
            print("RMSE (for daily level prediction) : " + str(rms))
            print("------------------------------------")
            hist_perf_list.append(perf_dict.copy())
    return pd.DataFrame(hist_perf_list).iloc[::-1]


def get_dates(start_date, no_days):
    return pd.date_range(start_date, periods=no_days+1).tolist()[1:]


def get_live_predictions(data, prediction_period=7):
    print("\n##### Generating live predictions for next  " + str(prediction_period) + " days #########\n")
    model = get_arima_model(data['no_cases'])
    predictions = get_forecast(model, prediction_period)
    return predictions


def predict_covid_cases(country, forecast_period=7, is_history=False):
    recent_predicted_date = get_recent_predicted_date(country)
    print("********************************",recent_predicted_date)
    if recent_predicted_date:
        print("Retrieving predictions from pickle file generate on : {}".format(recent_predicted_date))
        prediction_df, hist_df = get_prediction_from_file(country, recent_predicted_date)

    else:
        print("Pickle file not found {0}. Creating fresh predictions".format(recent_predicted_date))
        prediction_df, hist_df = get_fresh_predictions(country,
                                                       forecast_period=forecast_period,
                                                       is_history=is_history)
    return prediction_df, hist_df


def get_recent_predicted_date(country):
    country_folder_path = os.path.join(cwd,'predictions_store/'+country+ '/')
    print(country_folder_path)
    if os.path.isdir(country_folder_path) and len(os.listdir(country_folder_path)) > 0:
        files = os.listdir(country_folder_path)
        dates = [s.split('.pkl')[0] for s in files]
        dates.sort()
        return dates[-1]
    else:
        return None


def get_fresh_predictions(country, forecast_period=7, is_history=False):
    data = get_data(country)
    hist_df = pd.DataFrame()
    prediction_df = pd.DataFrame()
    if len(data) > 70:
        if is_history and len(data) > 100:
            hist_df = get_historical_performance(data, forecast_period=forecast_period)
        if len(hist_df) > 0:
            hist_df = hist_df[['train_period', 'test_period', 'actual_cases', 'predicted_cases', 'RMSE', 'R square']]
        predictions = get_live_predictions(data, prediction_period=forecast_period)
        prediction_df['date'] = get_dates(data['date'][-1], forecast_period)
        prediction_df['predicted_cases'] = predictions
    return prediction_df, hist_df


def get_prediction_from_file(country, pred_date):
    country_folder_path = os.path.join(cwd,'predictions_store/'+country+ '/')
    print(country_folder_path)
    with open(country_folder_path + pred_date + '.pkl', 'rb') as handle:
        pred_dict = pickle.load(handle)
        prediction_df = pred_dict['predictions']
        hist_df = pred_dict['historical_predictions']
        return prediction_df, hist_df


def batch_predict(country_list=None):
    if not country_list:
        country_list = covid_kaggle_data['Country/Region'].unique()
    for this_country in country_list:
        print("Starting batch predict for {}".format(this_country))
        country_folder_path = 'covid_19_django_ui/predictions_store/' + this_country + '/'
        if not os.path.isdir(country_folder_path):
            os.mkdir(country_folder_path)
        this_pred_df, this_history_df = get_fresh_predictions(this_country, forecast_period=7, is_history=True)
        store_dict = dict()
        store_dict['created_date'] = datetime.date.today()
        store_dict['predictions'] = this_pred_df.copy()
        store_dict['historical_predictions'] = this_history_df.copy()
        with open(country_folder_path + str(datetime.date.today()) + '.pkl', 'wb') as handle:
            pickle.dump(store_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Completed batch predict for {}".format(this_country))


