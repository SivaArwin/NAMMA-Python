# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 16:02:06 2018

@author: madis
"""

# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# reading a csv file using pandas library
wcat=pd.read_csv("E:\\Datasets\\Simple Linear Regression\\wc-at.csv")

wcat.columns

plt.hist(wcat.Waist)
plt.boxplot(wcat.Waist)
plt.plot(wcat.Waist,wcat.AT,"ro");plt.xlabel("waist");plt.ylabel("AT")
plt.hist(wcat.AT)
plt.boxplot(wcat.AT)


wcat.corr()
wcat.AT.corr(wcat.Waist) # # correlation value between X and Y
np.corrcoef(wcat.AT,wcat.Waist)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("AT~Waist",data=wcat).fit()
type(model)
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(wcat) # Predicted values of AT using the model

# Visualization of regresion line over the scatter plot of Waist and AT
# For visualization we need to import matplotlib.pyplot
import matplotlib.pyplot as plt
plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='red');plt.plot(wcat['Waist'],pred,color='black');plt.xlabel('WAIST');plt.ylabel('TISSUE')

pred.corr(wcat.AT) # 0.81
# Transforming variables for accuracy
model2 = smf.ols('AT~np.log(Waist)',data=wcat).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(wcat)
pred2.corr(wcat.AT)
# pred2 = model2.predict(wcat.iloc[:,0])
pred2
plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green');plt.plot(wcat['Waist'],pred2,color='blue');plt.xlabel('WAIST');plt.ylabel('TISSUE')

# Exponential transformation
model3 = smf.ols('np.log(AT)~Waist',data=wcat).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(wcat)
pred_log
pred3=np.exp(pred_log)  # as we have used log(AT) in preparing model so we need to convert it back
pred3
pred3.corr(wcat.AT)
plt.scatter(x=wcat['Waist'],y=wcat['AT'],color='green');plt.plot(wcat.Waist,np.exp(pred_log),color='blue');plt.xlabel('WAIST');plt.ylabel('TISSUE')
resid_3 = pred3-wcat.AT

# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(pred3,model3.resid_pearson,"o");plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=wcat.AT);plt.xlabel("Predicted");plt.ylabel("Actual")

# Quadratic model
wcat["Waist_Sq"] = wcat.Waist*wcat.Waist
model_quad = smf.ols("AT~Waist+Waist_Sq",data=wcat).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(wcat)

model_quad.conf_int(0.05) # 
plt.scatter(wcat.Waist,wcat.AT,c="b");plt.plot(wcat.Waist,pred_quad,"r")

plt.scatter(np.arange(109),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

plt.hist(model_quad.resid_pearson) # histogram for residual values 




