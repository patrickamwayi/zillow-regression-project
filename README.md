## Zillow Regression Project

## Predicting Tax Assessed Value
<hr style="border-top: 50px groove green; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Goals
>- The main goal of this project is to identify key drivers of Tax assessement for single family houses that were sold in 2017.

#### Project Description
>- Create a model that best predicts home values in the housing market.
>- Compare all models and evaluated by how well they performs over the baseline.


#### Initial Questions
>- 1. Does the number of bedrooms & bathrooms in a house influence the price of the home?
>- 2. Does size/area of the house influence the price of the home?
>- 3. Does the age of the house influence the price of the home?
>- 4. Does the location of the house influence the price of the home?

#### Audience
> - Codeup Data Science students

#### Project Deliverables
> - A final report notebook 
> - A final report notebook presentation
> - All necessary modules to make my project reproducible

#### Project Context
> - Zillow dataset from the Codeup database.


#### Data Dictionary

        Column                             Description                                              Data type
        
        bedroomcnt                         Number of bedrooms a house has                               Float

        bathroomcnt                        number of bathrooms a house has                              Float 
        
        calculatedfinishedsquarefeet       Size/Area in sq ft of the house                              Float 
        
        taxvaluedollarcnt                  Property tax value                                           Float 
        
        yearbuilt                          The year the property was built                              Float
        
        fips                               codes that are used to uniquely identify geographic areas    Float  
        
        
                                        
---     ------                                 --------------                                                 -----  

<hr style="border-top: 10px groove green; margin-top: 1px; margin-bottom: 1px"></hr>


### Initial Hypothesis

> - H0 : There is no relationship between number of bedrooms & bathrooms to home prices
> - Ha : There is a relationship between number of bedrooms & bathrooms to home prices
> - alpha = 0.05
> - x = dependent variable eg (train.bedrooms)
> - y = independent variable (train.assessed_value)

> - corr, p = stats.pearsonr(x, y)
> - print(f'corr = {corr:.5f}')
> - print(f'p = {p:.5f}')

> - if p<a:
> -    print(f"reject the null hypothesis")
> - else:
> -    print(f"reject the null hypothesis")
<hr style="border-top: 10px groove green; margin-top: 1px; margin-bottom: 1px"></hr>

### Conclusions & Next Steps
### Executive Summary 
<hr style="border-top: 10px groove green; margin-top: 1px; margin-bottom: 1px"></hr>

>- All the features I used in my exploration to predict home values were important especially size of homes. 
>- The number of bedrooms was surprisingly not as important as expected.
>- All the models that I used performed better than the baseline.
>- Of the top three models, Polynomial model is the best.

### Recommendations
>- I recommend that Zillow should use Polynomial Regression model to predict tax assessement value
>- Test data predicted an RMSE of 222,165.50 which is 38,622.50 more accurate than the baseline

### Next steps
> - Given more time I would include more columns on my dataset, improve my feature engineering, and increase model performance
<hr style="border-top: 10px groove green; margin-top: 1px; margin-bottom: 1px"></hr>

### Pipeline Stages Breakdown

<hr style="border-top: 10px groove green; margin-top: 1px; margin-bottom: 1px"></hr>

##### Plan
- [x] Create README.md with data dictionary, project and business goals, come up with initial hypotheses.
- [x] Acquire data from the Codeup Database and create a function to automate this process. Save the function in an acquire.py file to import into the Final Report Notebook.
- [x] Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process, store the function in a prepare.py module, and prepare data in Final Report Notebook by importing and using the funtion.
- [x]  Clearly define two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- [x] Establish a baseline accuracy and document well.
- [x] Train three different classification models.
- [x] Evaluate models on train and validate datasets.
- [x] Choose the model with that performs the best and evaluate that single model on the test dataset.
- [x] Create csv file with the measurement id, the probability of the target values, and the model's prediction for each observation in my test dataset.
- [x] Document conclusions, takeaways, and next steps in the Final Report Notebook.

___

##### Plan -> Acquire
> - Store functions that are needed to acquire data from the measures and species tables from Zillow database on the Codeup data science database server; make sure the acquire.py module contains the necessary imports to run my code.
> - The final function will return a pandas DataFrame.
> - Import the acquire function from the acquire.py module and use it to acquire the data in the Final Report Notebook.
> - Complete some initial data summarization (`.info()`, `.describe()`, `.value_counts()`, ...).
> - Plot distributions of individual variables.
___

##### Plan -> Acquire -> Prepare
> - Store functions needed to prepare the iris data; make sure the module contains the necessary imports to run the code. The final function should do the following:
    - Split the data into train/validate/test.
    - Handle any missing values.
    - Handle erroneous data and/or outliers that need addressing.
    - Encode variables as needed.
    - Create any new features, if made for this project.
> - Import the prepare function from the prepare.py module and use it to prepare the data in the Final Report Notebook.
___

##### Plan -> Acquire -> Prepare -> Explore
> - Answer key questions, my hypotheses, and figure out the features that can be used in a classification model to best predict the target variable, species. 
> - Run at least 2 statistical tests in data exploration. Document my hypotheses, set an alpha before running the tests, and document the findings well.
> - Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify features that are related to species (the target), identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.
> - Summarize my conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model
> - Establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 3 different models. Document these steps well.
> - Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.
> - Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.
> - Feature Selection (after initial iteration through pipeline): Are there any variables that seem to provide limited to no additional information? If so, remove them.
> - Based on the evaluation of the models using the train and validate datasets, choose the best model to try with the test data, once.
> - Test the final model on the out-of-sample data (the testing dataset), summarize the performance, interpret and document the results.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model -> Deliver
> - Introduce myself and my project goals at the very beginning of my notebook walkthrough.
> - Summarize my findings at the beginning like I would for an Executive Summary. (Don't throw everything out that I learned from Storytelling) .
> - Walk Codeup Data Science Team through the analysis I did to answer my questions and that lead to my findings. (Visualize relationships and Document takeaways.) 
> - Clearly call out the questions and answers I am analyzing as well as offer insights and recommendations based on my findings.

<hr style="border-top: 10px groove green; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproduce My Project

<hr style="border-top: 10px groove green; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download the aquire.py, prepare.py, and final_report.ipynb files into your working directory
- [ ] Add your own env file to your directory. (user, password, host)
- [ ] Run the final_report.ipynb notebook
