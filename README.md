# Spotify-song-Valence-prediction

In this project, we predict the Valence of a playlist of songs from Spotify. Data was taken from Spotify API using spotipy python library. 

Received the client credentials as well as the laylist_link from ''https://github.com/clarissaache/ml-spotify-song-genre-prediction#''. (These are made public in the repo).

I took the features artist_pop, track_pop, danceability, energy, key, loudness. mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms and time_signatue. Out of these, we predict valence. I decided to remove the rest of the features. 

After training the data (80:20 split), using Multiple linear regression, Support vector regression, Ridge regression, Lasso regression, Decision tree regression and Random forest regression, we predict the output. ('poly' kernel used in SVR; tried with others as well, but poly gave best results. Gave Number of estimators in random forest as 50). 

The error metrics Mean squared error, Mean absolute error, Coefficient of R2, and Root Mean Square error were used to test the output. Results are shown in the resukts file. We can see that all of the methods performed well, except for possibly, Multiple Linear Regression. The best outputs were given by Ridge and Random forest regression methods.
