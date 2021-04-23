# BT5153-Recommendation-System-for-Stack-Overflow
<br />

<p align="center">
  <a href="https://github.com/Benjaminlxl/BT5153-Recommendation-System-for-Stack-Overflow">
    <img src="images/logo.png" alt="Logo">
  </a>

<h3 align="center">Shoot the Long Waiting Pain — Recommendation System for Stack Overflow</h3>

  <p align="center">
    Group15 Project for NUS BT5153 FY2021 Spring 
  </p>

</p>



# About the Project

***In the project, we wish to predict the waiting time to receive a solid answer by identifying potential features.*** 

We have extracted a raw dataset from the Stack Exchange Data Explorer, a tool for executing arbitrary SQL queries against data from [Stack Overflow Database](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede), using the custom [query](https://data.stackexchange.com/stackoverflow/query/edit/1373717). This raw dataset consists of total 190234 questions which are relevant to Python and Data Science (post tags contain ‘python’, ‘numpy’ and ‘scikit-learn’ etc.) with the accepted answers in the past 2 years covered from Jan 2019 to Jan 2021. 

***By digging into the post content, we also wish to launch a recommendation system that provides the most relevant and popular posts to one certain question posted to improve user experience by shortening their waiting time.*** 



# Getting Started

## Dataset

## Code

- `src/waiting_time_prediction.ipynb`

  Primary notebook of waiting time prediction, including *data loading*, *data processing*, *model implementation* and *machine learning interpretation*.

- `src/waiting_time_prediction_auxiliary.ipynb`

  Auxiliary notebook of waiting time prediction which consists of *baseline model* without any data precessing.

- `src/recommendation_system.ipynb`

  Notebook of *recommendation system* powered by vector space matching technology



# Road-map

## Data Processing



## Feature Extraction



## Modeling

### Waiting Time Prediction

### Recommendation System Building



## Conclusion