# Spotify-song-Valence-prediction

In this project, we predict the Valence of a playlist of songs from Spotify. Data was taken from Spotify API using spotipy python library. 

Received the client credentials as well as the laylist_link from ''https://github.com/clarissaache/ml-spotify-song-genre-prediction#''. (These are made public in the repo).

I took the features artist_pop, track_pop, danceability, energy, key, loudness. mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms and time_signatue. Out of these, we predict valence. I decided to remove the rest of the features. 

After training the data (80:20 split), using Multiple linear regression, Support vector regression, Ridge regression, Lasso regression, Decision tree regression and Random forest regression, we predict the output. ('poly' kernel used in SVR; tried with others as well, but poly gave best results. Gave Number of estimators in random forest as 50). 

The error metrics Mean squared error, Mean absolute error, Coefficient of R2, and Root Mean Square error were used to test the output. Results are shown below - 

              Regressions       mse       mae        r2      rmse
0         Multiple Linear  0.238587  0.189680 -4.488255  0.488454
1  Support Vector Machine  0.036689  0.154123  0.156037  0.488454
2                   Ridge  0.015243  0.101427  0.649355  0.123464
3                   Lasso  0.036660  0.153130  0.156707  0.191468
4           Decision Tree  0.069880  0.212700 -0.607453  0.264347
5           Random Forest  0.023476  0.131676  0.459972  0.153220

All of the methods performed well, except for possibly, Multiple Linear Regression. The best outputs were given by Ridge and Random forest regression methods.
