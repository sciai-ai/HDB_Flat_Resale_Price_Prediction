# HDB Flat Resale Price Prediction


## Overview of Project Components

The following are a brief explanation of the different folders, document and files in this project repository:

## 1. Report - Resale Price Prediction of HDB Resale Flats in Singapore
- This is the full write-up report for the project.
- The purpose of the report is to provide a succint update of the entire project to project stakeholders.

## 2. Notebook - HDB Flat Resale Price Prediction
- Jupyter Notebook that contains the full codes of the project and detailed explanation and comments for each step of the process.
- The notebook is intended for users and collaborators to access the full codes of the project.

## 3. Folder - Data
- This folder contains all the datasets used in this project.
- The various datasets are saved in this folder to reduce resources required to extract the full data from various APIs.
  - dat_hdb.csv: HDB Flat Resale Transaction Data (Jan 2017 - Jan 2020), retrieved from Data.gov.sg
  - dat_coord.csv: Coordinate data for each transaction address extracted via geocoding
  - dat_venues.csv: Nearby venues data for each transaction location within 500m radius, extracted from Foursquare
  - dat_results.csv: Summary table of the performance metrics between the two models
  - dat_pred.csv: Prediction results vs actual transaction price of HDB Resale Flats.
  - dat_final.csv: Processed data used for model building.

## 4. Folder - Images
- This folder contains all the graphs used in the Presentation Deck and Report.
- Codes to generate the graphs are recorded in the Notebook.

## 5. Folder - Models
- This folder contains the saved model developed for the project.
- The final model saved is used for production deployment as part of an application.
