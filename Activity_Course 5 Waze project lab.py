#!/usr/bin/env python
# coding: utf-8

# # **Waze Project**
# **Course 5 - Regression analysis: Simplify complex data relationships**

# Your team is more than halfway through their user churn project. Earlier, you completed a project proposal, used Python to explore and analyze Waze’s user data, created data visualizations, and conducted a hypothesis test. Now, leadership wants your team to build a regression model to predict user churn based on a variety of variables.
# 
# You check your inbox and discover a new email from Ursula Sayo, Waze's Operations Manager. Ursula asks your team about the details of the regression model. You also notice two follow-up emails from your supervisor, May Santner. The first email is a response to Ursula, and says that the team will build a binomial logistic regression model. In her second email, May asks you to help build the model and prepare an executive summary to share your results.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions and prepare an executive summary.

# # **Course 5 End-of-course project: Regression modeling**
# 
# In this activity, you will build a binomial logistic regression model. As you have learned, logistic regression helps you estimate the probability of an outcome. For data science professionals, this is a useful skill because it allows you to consider more than one variable against the variable you're measuring against. This opens the door for much more thorough and flexible analysis to be completed.
# <br/>
# 
# **The purpose** of this project is to demostrate knowledge of exploratory data analysis (EDA) and a binomial logistic regression model.
# 
# **The goal** is to build a binomial logistic regression model and evaluate the model's performance.
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** EDA & Checking Model Assumptions
# * What are some purposes of EDA before constructing a binomial logistic regression model?
# 
# **Part 2:** Model Building and Evaluation
# * What resources do you find yourself using as you complete this stage?
# 
# **Part 3:** Interpreting Model Results
# 
# * What key insights emerged from your model(s)?
# 
# * What business recommendations do you propose based on the models built?
# 
# <br/>
# 
# Follow the instructions and answer the question below to complete the activity. Then, you will complete an executive summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.

# # **Build a regression model**

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="../Images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.

# ### **Task 1. Imports and data loading**
# Import the data and packages that you've learned are needed for building logistic regression models.

# In[44]:


# Packages for numerics + dataframes
### YOUR CODE HERE ###
import pandas as pd
import numpy as np

# Packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Packages for Logistic Regression & Confusion Matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

# Packages for visualization
### YOUR CODE HERE ###

# Packages for Logistic Regression & Confusion Matrix
### YOUR CODE HERE ###


# Import the dataset.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[45]:


# Load the dataset by running this cell
df = pd.read_csv('https://raw.githubusercontent.com/adacert/waze/main/Synthetic_Waze_Data_14999%20-%20Fictional_Waze_Data_14999.csv')


# <img src="../Images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.
# 
# In this stage, consider the following question:
# 
# * What are some purposes of EDA before constructing a binomial logistic regression model?

# ==> ENTER YOUR RESPONSE HERE

# ### **Task 2a. Explore data with EDA**
# 
# Analyze and discover data, looking for correlations, missing data, potential outliers, and/or duplicates.
# 
# 

# Start with `.shape` and `info()`.

# In[46]:


### YOUR CODE HERE ###
df.shape
df.info()


# **Question:** Are there any missing values in your data?

# ==> ENTER YOUR RESPONSE HERE

# Use `.head()`.
# 
# 

# In[47]:


### YOUR CODE HERE ###
df.head()


# Use `.drop()` to remove the ID column since we don't need this information for your analysis.

# In[48]:


### YOUR CODE HERE ###
df = df.drop('ID',axis = 1)


# Now, check the class balance of the dependent (target) variable, `label`.

# In[49]:


### YOUR CODE HERE ###
df.describe()


# Call `.describe()` on the data.
# 

# In[50]:


### YOUR CODE HERE ###


# **Question:** Are there any variables that could potentially have outliers just by assessing at the quartile values, standard deviation, and max values?

# ==> ENTER YOUR RESPONSE HERE

# ### **Task 2b. Create features**
# 
# Create features that may be of interest to the stakeholder and/or that are needed to address the business scenario/problem.

# #### **`km_per_driving_day`**
# 
# You know from earlier EDA that churn rate correlates with distance driven per driving day in the last month. It might be helpful to engineer a feature that captures this information.
# 
# 1. Create a new column in `df` called `km_per_driving_day`, which represents the mean distance driven per driving day for each user.
# 
# 2. Call the `describe()` method on the new column.

# In[51]:


# 1. Create `km_per_driving_day` column
### YOUR CODE HERE ###
df['km_per_driving_day'] =df['driven_km_drives'] / df['driving_days']
# 2. Call `describe()` on the new column
df['km_per_driving_day'].describe()
### YOUR CODE HERE ###


# Note that some values are infinite. This is the result of there being values of zero in the `driving_days` column. Pandas imputes a value of infinity in the corresponding rows of the new column because division by zero is undefined.
# 
# 1. Convert these values from infinity to zero. You can use `np.inf` to refer to a value of infinity.
# 
# 2. Call `describe()` on the `km_per_driving_day` column to verify that it worked.

# #### **`professional_driver`**
# 
# Create a new, binary feature called `professional_driver` that is a 1 for users who had 100 or more drives <u>**and**</u> drove on 20+ days in the last month.
# 
# **Note:** The objective is to create a new feature that separates professional drivers from other drivers. In this scenario, domain knowledge and intuition are used to determine these deciding thresholds, but ultimately they are arbitrary.

# In[52]:


# 1. Convert infinite values to zero
### YOUR CODE HERE ###
df.loc[df['km_per_driving_day']== np.inf,'km_per_driving_day'] = 0
# 2. Confirm that it worked
df['km_per_driving_day'].describe()
### YOUR CODE HERE ###


# To create this column, use the [`np.where()`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) function. This function accepts as arguments:
# 1. A condition
# 2. What to return when the condition is true
# 3. What to return when the condition is false
# 
# ```
# Example:
# x = [1, 2, 3]
# x = np.where(x > 2, 100, 0)
# x
# array([  0,   0, 100])
# ```

# In[53]:


# Create `professional_driver` column
df['professional_driver'] = np.where((df['drives']>= 100) & (df['driving_days'] > 20), 1, 0)
### YOUR CODE HERE ###


# Perform a quick inspection of the new variable.
# 
# 1. Check the count of professional drivers and non-professionals
# 
# 2. Within each class (professional and non-professional) calculate the churn rate

# In[54]:


# 1. Check count of professionals and non-professionals
### YOUR CODE HERE ###
df['professional_driver'].value_counts()
# 2. Check in-class churn rate
df.groupby(['professional_driver'])['label'].value_counts(normalize=True)
### YOUR CODE HERE ###


# The churn rate for professional drivers is 7.6%, while the churn rate for non-professionals is 19.9%. This seems like it could add predictive signal to the model.

# <img src="../Images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# 
# After analysis and deriving variables with close relationships, it is time to begin constructing the model.
# 
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.
# 
# In this stage, consider the following question:
# 
# * Why did you select the X variables you did?

# ==> ENTER YOUR RESPONSE HERE

# ### **Task 3a. Preparing variables**

# Call `info()` on the dataframe to check the data type of the `label` variable and to verify if there are any missing values.

# In[55]:


### YOUR CODE HERE ###
df.info()


# Because you know from previous EDA that there is no evidence of a non-random cause of the 700 missing values in the `label` column, and because these observations comprise less than 5% of the data, use the `dropna()` method to drop the rows that are missing this data.

# In[56]:


# Drop rows with missing data in `label` column
df['label'].dropna()
### YOUR CODE HERE ###


# #### **Impute outliers**
# 
# You rarely want to drop outliers, and generally will not do so unless there is a clear reason for it (e.g., typographic errors).
# 
# At times outliers can be changed to the **median, mean, 95th percentile, etc.**
# 
# Previously, you determined that seven of the variables had clear signs of containing outliers:
# 
# * `sessions`
# * `drives`
# * `total_sessions`
# * `total_navigations_fav1`
# * `total_navigations_fav2`
# * `driven_km_drives`
# * `duration_minutes_drives`
# 
# For this analysis, impute the outlying values for these columns. Calculate the **95th percentile** of each column and change to this value any value in the column that exceeds it.
# 

# In[57]:


# Impute outliers
for column in ['sessions', 'drives', 'total_sessions', 'total_navigations_fav1',
               'total_navigations_fav2', 'driven_km_drives', 'duration_minutes_drives']:
    threshold = df[column].quantile(0.95)
    df.loc[df[column]>threshold,column] = threshold
    
### YOUR CODE HERE ###


# Call `describe()`.

# In[58]:


### YOUR CODE HERE ###
df.describe()


# #### **Encode categorical variables**

# Change the data type of the `label` column to be binary. This change is needed to train a logistic regression model.
# 
# Assign a `0` for all `retained` users.
# 
# Assign a `1` for all `churned` users.
# 
# Save this variable as `label2` as to not overwrite the original `label` variable.
# 
# **Note:** There are many ways to do this. Consider using `np.where()` as you did earlier in this notebook.

# In[59]:


# Create binary `label2` column
df['label2'] = np.where(df['label']=='churned',1,0)
df[['label', 'label2']].tail()
### YOUR CODE HERE ###


# ### **Task 3b. Determine whether assumptions have been met**
# 
# The following are the assumptions for logistic regression:
# 
# * Independent observations (This refers to how the data was collected.)
# 
# * No extreme outliers
# 
# * Little to no multicollinearity among X predictors
# 
# * Linear relationship between X and the **logit** of y
# 
# For the first assumption, you can assume that observations are independent for this project.
# 
# The second assumption has already been addressed.
# 
# The last assumption will be verified after modeling.
# 
# **Note:** In practice, modeling assumptions are often violated, and depending on the specifics of your use case and the severity of the violation, it might not affect your model much at all or it will result in a failed model.

# #### **Collinearity**
# 
# Check the correlation among predictor variables. First, generate a correlation matrix.

# In[60]:


# Generate a correlation matrix
df.corr(method='pearson')
### YOUR CODE HERE ###


# Now, plot a correlation heatmap.

# In[61]:


# Plot correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr('pearson'), vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
### YOUR CODE HERE ###


# If there are predictor variables that have a Pearson correlation coefficient value greater than the **absolute value of 0.7**, these variables are strongly multicollinear. Therefore, only one of these variables should be used in your model.
# 
# **Note:** 0.7 is an arbitrary threshold. Some industries may use 0.6, 0.8, etc.
# 
# **Question:** Which variables are multicollinear with each other?

# ==> ENTER YOUR RESPONSE HERE

# ### **Task 3c. Create dummies (if necessary)**
# 
# If you have selected `device` as an X variable, you will need to create dummy variables since this variable is categorical.
# 
# In cases with many categorical variables, you can use pandas built-in [`pd.get_dummies()`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), or you can use scikit-learn's [`OneHotEncoder()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) function.
# 
# **Note:** Variables with many categories should only be dummied if absolutely necessary. Each category will result in a coefficient for your model which can lead to overfitting.
# 
# Because this dataset only has one remaining categorical feature (`device`), it's not necessary to use one of these special functions. You can just implement the transformation directly.
# 
# Create a new, binary column called `device2` that encodes user devices as follows:
# 
# * `Android` -> `0`
# * `iPhone` -> `1`

# In[62]:


# Create new `device2` variable
df['device2'] = np.where(df['device']=='Android', 0, 1)
df[['device', 'device2']].tail()
### YOUR CODE HERE ###


# ### **Task 3d. Model building**

# #### **Assign predictor variables and target**
# 
# To build your model you need to determine what X variables you want to include in your model to predict your target&mdash;`label2`.
# 
# Drop the following variables and assign the results to `X`:
# 
# * `label` (this is the target)
# * `label2` (this is the target)
# * `device` (this is the non-binary-encoded categorical variable)
# * `sessions` (this had high multicollinearity)
# * `driving_days` (this had high multicollinearity)
# 
# **Note:** Notice that `sessions` and `driving_days` were selected to be dropped, rather than `drives` and `activity_days`. The reason for this is that the features that were kept for modeling had slightly stronger correlations with the target variable than the features that were dropped.

# In[63]:


# Isolate predictor variables
X = df.drop(columns = ['label', 'label2', 'device', 'sessions', 'driving_days'])
### YOUR CODE HERE ###


# Now, isolate the dependent (target) variable. Assign it to a variable called `y`.

# In[64]:


# Isolate target variable
y = df['label2']
### YOUR CODE HERE ###


# #### **Split the data**
# 
# Use scikit-learn's [`train_test_split()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function to perform a train/test split on your data using the X and y variables you assigned above.
# 
# **Note 1:** It is important to do a train test to obtain accurate predictions.  You always want to fit your model on your training set and evaluate your model on your test set to avoid data leakage.
# 
# **Note 2:** Because the target class is imbalanced (82% retained vs. 18% churned), you want to make sure that you don't get an unlucky split that over- or under-represents the frequency of the minority class. Set the function's `stratify` parameter to `y` to ensure that the minority class appears in both train and test sets in the same proportion that it does in the overall dataset.

# In[65]:


# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y,random_state = 42)
### YOUR CODE HERE ###


# In[66]:


# Use .head()
### YOUR CODE HERE ###
X_train.head()


# Use scikit-learn to instantiate a logistic regression model. Add the argument `penalty = None`.
# 
# It is important to add `penalty = None` since your predictors are unscaled.
# 
# Refer to scikit-learn's [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) documentation for more information.
# 
# Fit the model on `X_train` and `y_train`.

# In[67]:


### YOUR CODE HERE ###
model = LogisticRegression(penalty = 'none', max_iter = 400)
model.fit(X_train, y_train)


# Call the `.coef_` attribute on the model to get the coefficients of each variable.  The coefficients are in order of how the variables are listed in the dataset.  Remember that the coefficients represent the change in the **log odds** of the target variable for **every one unit increase in X**.
# 
# If you want, create a series whose index is the column names and whose values are the coefficients in `model.coef_`.

# In[68]:


### YOUR CODE HER
pd.Series(model.coef_[0],index = X.columns)


# Call the model's `intercept_` attribute to get the intercept of the model.

# In[69]:


### YOUR CODE HERE ###
model.intercept_


# #### **Check final assumption**
# 
# Verify the linear relationship between X and the estimated log odds (known as logits) by making a regplot.
# 
# Call the model's `predict_proba()` method to generate the probability of response for each sample in the training data. (The training data is the argument to the method.) Assign the result to a variable called `training_probabilities`. This results in a 2-D array where each row represents a user in `X_train`. The first column is the probability of the user not churning, and the second column is the probability of the user churning.

# In[70]:


# Get the predicted probabilities of the training data
training_probabilities = model.predict_proba(X_train)
training_probabilities
### YOUR CODE HERE ###


# In logistic regression, the relationship between a predictor variable and the dependent variable does not need to be linear, however, the log-odds (a.k.a., logit) of the dependent variable with respect to the predictor variable should be linear. Here is the formula for calculating log-odds, where _p_ is the probability of response:
# <br>
# $$
# logit(p) = ln(\frac{p}{1-p})
# $$
# <br>
# 
# 1. Create a dataframe called `logit_data` that is a copy of `df`.
# 
# 2. Create a new column called `logit` in the `logit_data` dataframe. The data in this column should represent the logit for each user.
# 

# In[71]:


### YOUR CODE HERE ###
logit_data = X_train.copy()
logit_data['logit'] = [np.log(prob[1]/prob[0]) for prob in training_probabilities]


# In logistic regression, the relationship between a predictor variable and the dependent variable does not need to be linear, however, the log-odds (a.k.a., logit) of the dependent variable with respect to the predictor variable should be linear. Here is the formula for calculating log-odds, where _p_ is the probability of response:
# <br>
# $$
# logit(p) = ln(\frac{p}{1-p})
# $$
# <br>
# 
# 1. Create a dataframe called `logit_data` that is a copy of `df`.
# 
# 2. Create a new column called `logit` in the `logit_data` dataframe. The data in this column should represent the logit for each user.
# 

# In[72]:


# 1. Copy the `X_train` dataframe and assign to `logit_data`
### YOUR CODE HERE ###
logit_data = X_train.copy()

# 2. Create a new `logit` column in the `logit_data` df
### YOUR CODE HERE ###
logit_data['logit'] = [np.log(prob[1]/prob[0]) for prob in training_probabilities]


# Plot a regplot where the x-axis represents an independent variable and the y-axis represents the log-odds of the predicted probabilities.
# 
# In an exhaustive analysis, this would be plotted for each continuous or discrete predictor variable. Here we show only `driving_days`.

# In[73]:


# Plot regplot of `activity_days` log-odds
sns.regplot(x='activity_days', y='logit', data=logit_data, scatter_kws={'s': 2, 'alpha': 0.5})
plt.title('Log-odds: activity_days');
### YOUR CODE HERE ###


# <img src="../Images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 4a. Results and evaluation**
# 
# If the logistic assumptions are met, the model results can be appropriately interpreted.
# 
# Use the code block below to make predictions on the test data.
# 

# In[74]:


# Generate predictions on X_test
y_preds = model.predict(X_test)
### YOUR CODE HERE ###


# Now, use the `score()` method on the model with `X_test` and `y_test` as its two arguments. The default score in scikit-learn is **accuracy**.  What is the accuracy of your model?
# 
# *Consider:  Is accuracy the best metric to use to evaluate this model?*

# In[75]:


# Score the model (accuracy) on the test data
### YOUR CODE HERE ###
model.score(X_test,y_test)


# ### **Task 4b. Show results with a confusion matrix**

# Use the `confusion_matrix` function to obtain a confusion matrix. Use `y_test` and `y_preds` as arguments.

# In[76]:


### YOUR CODE HERE ###
cm = confusion_matrix(y_test, y_preds)


# Next, use the `ConfusionMatrixDisplay()` function to display the confusion matrix from the above cell, passing the confusion matrix you just created as its argument.

# In[77]:


### YOUR CODE HERE
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels='none')
disp.plot();


# You can use the confusion matrix to compute precision and recall manually. You can also use scikit-learn's [`classification_report()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) function to generate a table from `y_test` and `y_preds`.

# In[78]:


# Calculate precision manually
precision = cm[1,1] / (cm[0, 1] + cm[1, 1])
precision
### YOUR CODE HERE ###


# In[79]:


# Calculate recall manually
recall = cm[1,1] / (cm[1,0] + cm[1,1])
recall
### YOUR CODE HERE ###


# In[82]:


# Create a classification report
### YOUR CODE HERE ###
target_labels = ['retained', 'churned']
print(classification_report(y_test, y_preds, target_names=target_labels))


# **Note:** The model has decent precision but very low recall, which means that it makes a lot of false negative predictions and fails to capture users who will churn.

# ### **BONUS**
# 
# Generate a bar graph of the model's coefficients for a visual representation of the importance of the model's features.

# In[87]:


feature_importance = list(zip(X_train.columns, model.coef_[0]))

# Sort the list by coefficient value
feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
feature_importance
# Sort the list by coefficient value
### YOUR CODE HERE ###


# In[88]:


# Plot the feature importances
import seaborn as sns
sns.barplot(x=[x[1] for x in feature_importance],
            y=[x[0] for x in feature_importance],
            orient='h')
### YOUR CODE HERE ###


# ### **Task 4c. Conclusion**
# 
# Now that you've built your regression model, the next step is to share your findings with the Waze leadership team. Consider the following questions as you prepare to write your executive summary. Think about key points you may want to share with the team, and what information is most relevant to the user churn project.
# 
# **Questions:**
# 
# 1. What variable most influenced the model's prediction? How? Was this surprising?
# 
# 2. Were there any variables that you expected to be stronger predictors than they were?
# 
# 3. Why might a variable you thought to be important not be important in the model?
# 
# 4. Would you recommend that Waze use this model? Why or why not?
# 
# 5. What could you do to improve this model?
# 
# 6. What additional features would you like to have to help improve the model?
# 

# 1) activity days had the most influence on the outcome variable of churn rate with a negative correlation. 
# 2)Yes. In previous EDA, user churn rate increased as the values in km_per_driving_day increased. The correlation heatmap here in this notebook revealed this variable to have the strongest positive correlation with churn of any of the predictor variables by a relatively large margin. In the model, it was the second-least-important variable.
# 3) the model eventhough has a strong precision score has a low recall score which would not make it a good model for making business decisions. 
# 4) 
