# Fare Game - Cracking the Code on Tips, Trips, and Taxi Profit 

## Research Questions
RQ1: How do time, location, and trip characteristics influence the likelihood of passengers tipping?

RQ2: How do different trip features (time, distance, fare, location) influence tipping behavior?

RQ3: Based on tipping and demand patterns, what trip lengths are most profitable across different regions?

## Dataset

Trip data downloaded from
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

We used the Yellow Taxi Trip Records from January 2023 - November 2024 (24 files total).

Once files are downloaded, run ```preprocessing.ipynb``` to get ```tripdata_combined.parquet```. Other notebooks rely on this file for analysis.

## Running the App Locally

1. Run ```git clone https://github.com/Aleyssu/CabBoost.git``` and CD into this repo's folder.
2. Run ```pip install -r requirements.txt```.
3. Run ```streamlit run cab_boost_app.py```

Note: the app relies on the ```app_data``` and ```models``` folder to deliver recommendations.  It is essential that both folders and their contents are present before running the app.

To create these folders manually, clone this repo, CD into the repo's folder, and run ```trip_profitability_analysis.ipynb```. 

## Running Jupyter Notebooks

1. Run ```git clone https://github.com/Aleyssu/CabBoost.git``` and CD into this repo's folder.
2. Run ```pip install -r requirements.txt```.

### RQ1:
- ```alex_rq1.ipynb``` 
- ```alex_rq1_model.pkl``` (Our best model)
### RQ2:
- ```RQ2regression_model.ipynb``` 
- ```kevin_rq2_work.ipynb``` 
### RQ3:
- ```vinny_rq3.ipynb``` 
- ```trip_profitability_analysis.py``` (Creates Analysis Data for App)
