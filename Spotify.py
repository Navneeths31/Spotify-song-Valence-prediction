import spotipy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


sp = spotipy.Spotify(client_credentials_manager = spotipy.oauth2.SpotifyClientCredentials(
    "e29de11330db48319de9c0b4cc239699","1d923d3159a2448989e8c11243c36a88"))

playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DWZBCPUIUs2iR?si=d9570de1add24fd0"

playlist_uri = playlist_link.split("/")[-1].split('?')[0]


dict = {
    "track_uri": [],
    "track_name": [],
    "artist_uri": [],
    "artist_info": [],
    "artist_name": [],
    "artist_pop": [],
    "artist_genres": [],
    "album": [],
    "track_pop": [],
}

for track in sp.playlist_tracks(playlist_uri)["items"]:
    # URI
    track_uri = track["track"]["uri"]
    dict["track_uri"].append(track_uri)
    # Track name
    track_name = track["track"]["name"]
    dict["track_name"].append(track_name)
    # Artist URI
    artist_uri = track["track"]["artists"][0]["uri"]
    dict["artist_uri"].append(artist_uri)  
    # Artist Info
    artist_info = sp.artist(artist_uri)
    dict["artist_info"].append(artist_info)
    # Name, popularity, genre
    artist_name = track["track"]["artists"][0]["name"]
    dict["artist_name"].append(artist_name)
    artist_pop = artist_info["popularity"]
    dict["artist_pop"].append(artist_pop)
    artist_genres = artist_info["genres"]
    dict["artist_genres"].append(artist_genres)
    # Album
    album = track["track"]["album"]["name"]
    dict["album"].append(album)
    # Popularity of the track
    track_pop = track["track"]["popularity"]
    dict["track_pop"].append(track_pop)
    
track_features = {
    "danceability": [],
    "energy": [],
    "key": [],
    "loudness": [],
    "mode": [],
    "speechiness": [],
    "acousticness": [],
    "instrumentalness": [],
    "liveness": [],
    "valence": [],
    "tempo": [],
    "id": [],
    "uri": [],
    "track_href": [],
    "analysis_url": [],
    "duration_ms": [],
    "time_signature": [],
}

for i in range(len(dict["track_uri"])):
    features = sp.audio_features(dict["track_uri"][i])[0]
    # print(features)
    # for features in audio_features:
    track_features["danceability"].append(features["danceability"])
    track_features["energy"].append(features["energy"])
    track_features["key"].append(features["key"])
    track_features["loudness"].append(features["loudness"])
    track_features["mode"].append(features["mode"])
    track_features["speechiness"].append(features["speechiness"])
    track_features["acousticness"].append(features["acousticness"])
    track_features["instrumentalness"].append(features["instrumentalness"])
    track_features["liveness"].append(features["liveness"])
    track_features["valence"].append(features["valence"])
    track_features["tempo"].append(features["tempo"])
    track_features["id"].append(features["id"])
    track_features["uri"].append(features["uri"])
    track_features["track_href"].append(features["track_href"])
    track_features["analysis_url"].append(features["analysis_url"])
    track_features["duration_ms"].append(features["duration_ms"])
    track_features["time_signature"].append(features["time_signature"])

df_track_data = pd.DataFrame.from_dict(dict)
df_track_features = pd.DataFrame.from_dict(track_features)
df_merged = pd.merge(
    df_track_data, df_track_features, how="inner", left_on="track_uri", right_on="uri")
print(df_merged)





df_merged = df_merged.drop(['track_uri','track_name','artist_uri','artist_info',
                            'artist_name','artist_genres','album','id','uri',
                            'track_href','analysis_url'], axis=1)




x = df_merged.drop("valence", axis=1)
y = np.array(df_merged["valence"])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
print(x_train, x_test, y_train, y_test)





linreg = linear_model.LinearRegression()
linreg.fit(x_train, y_train)
y_pred_lin = linreg.predict(x_test)
linreg_model_diff = pd.DataFrame({'Actual value':y_test, 'Predicted value': y_pred_lin})
print(linreg_model_diff)

linregmse = mean_squared_error(y_test, y_pred_lin)
linregmae = mean_absolute_error(y_test, y_pred_lin)
linregr2 = r2_score(y_test, y_pred_lin)
linregrmse = np.sqrt(linregmse)
print(linregmse, linregmae, linregr2, linregrmse)




svrreg = SVR(kernel='poly')
svrreg.fit(x_train,y_train)
y_pred_svr = svrreg.predict(x_test)
svrreg_model_diff = pd.DataFrame({'Actual value':y_test, 'Predicted value': y_pred_svr})
print(svrreg_model_diff)

svrregmse = mean_squared_error(y_test, y_pred_svr)
svrregmae = mean_absolute_error(y_test, y_pred_svr)
svrregr2 = r2_score(y_test, y_pred_svr)
svrregrmse = np.sqrt(linregmse)
print(svrregmse, svrregmae, svrregr2, svrregrmse)




ridreg = linear_model.Ridge(alpha=1.0)
ridreg.fit(x_train,y_train)
y_pred_rid = ridreg.predict(x_test)
ridreg_model_diff = pd.DataFrame({'Actual value':y_test, 'Predicted value': y_pred_rid})
print(ridreg_model_diff)

ridregmse = mean_squared_error(y_test, y_pred_rid)
ridregmae = mean_absolute_error(y_test, y_pred_rid)
ridregr2 = r2_score(y_test, y_pred_rid)
ridregrmse = np.sqrt(ridregmse)
print(ridregmse, ridregmae, ridregr2, ridregrmse)




lasreg = linear_model.Lasso()
lasreg.fit(x_train,y_train)
y_pred_las = lasreg.predict(x_test)
lasreg_model_diff = pd.DataFrame({'Actual value':y_test, 'Predicted value': y_pred_las})
print(lasreg_model_diff)

lasregmse = mean_squared_error(y_test, y_pred_las)
lasregmae = mean_absolute_error(y_test, y_pred_las)
lasregr2 = r2_score(y_test, y_pred_las)
lasregrmse = np.sqrt(lasregmse)
print(lasregmse, lasregmae, lasregr2, lasregrmse)



decreg = DecisionTreeRegressor()
decreg.fit(x_train,y_train)
y_pred_dec = decreg.predict(x_test)
decreg_model_diff = pd.DataFrame({'Actual value':y_test, 'Predicted value': y_pred_dec})
print(decreg_model_diff)

decregmse = mean_squared_error(y_test, y_pred_dec)
decregmae = mean_absolute_error(y_test, y_pred_dec)
decregr2 = r2_score(y_test, y_pred_dec)
decregrmse = np.sqrt(decregmse)
print(decregmse, decregmae, decregr2, decregrmse)




ranreg = RandomForestRegressor(n_estimators=50)
ranreg.fit(x_train,y_train)
y_pred_ran = ranreg.predict(x_test)
ranreg_model_diff = pd.DataFrame({'Actual value':y_test, 'Predicted value': y_pred_ran})
print(ranreg_model_diff)

ranregmse = mean_squared_error(y_test, y_pred_ran)
ranregmae = mean_absolute_error(y_test, y_pred_ran)
ranregr2 = r2_score(y_test, y_pred_ran)
ranregrmse = np.sqrt(ranregmse)
print(ranregmse, ranregmae, ranregr2, ranregrmse)





dict_metrics = {
'Regressions' : ['Multiple Linear', 'Support Vector Machine', 'Ridge', 'Lasso', 'Decision Tree', 'Random Forest'],
'mse' : [linregmse, svrregmse, ridregmse, lasregmse, decregmse, ranregmse],
'mae' : [linregmae, svrregmae, ridregmae, lasregmae, decregmae, ranregmae],
'r2' : [linregr2, svrregr2, ridregr2, lasregr2, decregr2, ranregr2],
'rmse' : [linregrmse, svrregrmse, ridregrmse, lasregrmse, decregrmse, ranregrmse]
}

dict_metrics = pd.DataFrame.from_dict(dict_metrics)
print(dict_metrics)