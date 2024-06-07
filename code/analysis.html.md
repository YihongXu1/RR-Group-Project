---
title: "Machine Learning Method Application to Covid-19 Dataset"
author: ""
date: today
format:
  html:
    toc: true
    toc-title: Contents
    toc-depth: 2
    toc-expand: 1
    smooth-scroll: true
    theme:
      light: lumen
      dark: superhero
number-sections: true
number-depth: 2
editor: visual
execute:
  warning: false
  cache: true 
keep-md: true
title-block-banner: true
---



# Preparation

Load all related packages, and define hyper-parameters from the very beginning. To ensure reproductivity, all seed are set to the fixed value `seed`.


::: {.cell}

```{.r .cell-code}
library(tidyverse) # data cleaning
library(knitr) # display tables in good format
library(caret) # confusion matrics
library(bnclassify) # BN
library(kernlab) # SVM
library(randomForest) # random forest
library(C50) # C5.0 decision tree
```
:::

::: {.cell}

```{.r .cell-code}
# set hyperparameters
path_csv <- "../Data/patient.csv"
seed <- 9
```
:::


# Data

## Loading

First of all, load the raw data. There are 95839 samples and 20 features (including the response variable) here.


::: {.cell}

```{.r .cell-code}
df <- read.csv(path_csv)
dim(df)
```

::: {.cell-output .cell-output-stdout}

```
[1] 95839    20
```


:::

```{.r .cell-code}
kable(head(df), format = "html")
```

::: {.cell-output-display}

`````{=html}
<table>
 <thead>
  <tr>
   <th style="text-align:right;"> sex </th>
   <th style="text-align:right;"> patient_type </th>
   <th style="text-align:right;"> intubated </th>
   <th style="text-align:right;"> pneumonia </th>
   <th style="text-align:right;"> age </th>
   <th style="text-align:right;"> pregnant </th>
   <th style="text-align:right;"> diabetes </th>
   <th style="text-align:right;"> copd </th>
   <th style="text-align:right;"> asthma </th>
   <th style="text-align:right;"> immunosuppression </th>
   <th style="text-align:right;"> hypertension </th>
   <th style="text-align:right;"> other_diseases </th>
   <th style="text-align:right;"> cardiovascular </th>
   <th style="text-align:right;"> obesity </th>
   <th style="text-align:right;"> chronic_kidney_failure </th>
   <th style="text-align:right;"> smoker </th>
   <th style="text-align:right;"> another_case </th>
   <th style="text-align:right;"> outcome </th>
   <th style="text-align:right;"> icu </th>
   <th style="text-align:left;"> death_date </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 42 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 99 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:left;"> 9999-99-99 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 51 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 99 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:left;"> 9999-99-99 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 51 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 99 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:left;"> 9999-99-99 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 57 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 99 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:left;"> 2020-04-01 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 44 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:left;"> 9999-99-99 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 40 </td>
   <td style="text-align:right;"> 97 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 98 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 99 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:right;"> 2 </td>
   <td style="text-align:left;"> 9999-99-99 </td>
  </tr>
</tbody>
</table>

`````

:::
:::


## Cleaning

Aiming to solve the binary classification problem, binary `y` should be elicited first. As for features, not all of them are available. For example, `pneumonia = 99` indicate the feature `pneumonia` is not available for this sample. In this case, we should label `99` as `NA` to avoid future mistakes. Some machine learning methods cannot handle cases with NA value, we actually use sample without NA values. however, NA for feature `pregnant` is not really not available. All males are labelled `NA`, but it does not make sense to eliminate all males. In this case, `NA` for feature `pregnant` should be changed to `2` which indicate not pregnant. Besides, `age` is the only continuous variable in this dataset. NaiveBayes and BayesNet could only handle factor features, while others work with numeric features. We prepare two version of data frames, one with factor age column and the other with numeric age column.


::: {.cell}

```{.r .cell-code}
df_ageori <- df %>% mutate(y = factor(ifelse(death_date == "9999-99-99", 0, 1),
                                      labels = c("live", "die")),
                           pregnant = factor(ifelse(pregnant == 1, 1, 2)),
              across(c(sex, patient_type, intubated, pneumonia,
                       diabetes, copd, asthma, immunosuppression, hypertension,
                       other_diseases, cardiovascular, obesity, chronic_kidney_failure,
                       smoker, another_case, icu, outcome), ~ factor(ifelse(.>2, NA, .)))) %>%
  select(-c(death_date))
df_agefac <- df_ageori %>% mutate(age = factor(age))
kable(head(df_agefac), format = "html")
```

::: {.cell-output-display}

`````{=html}
<table>
 <thead>
  <tr>
   <th style="text-align:left;"> sex </th>
   <th style="text-align:left;"> patient_type </th>
   <th style="text-align:left;"> intubated </th>
   <th style="text-align:left;"> pneumonia </th>
   <th style="text-align:left;"> age </th>
   <th style="text-align:left;"> pregnant </th>
   <th style="text-align:left;"> diabetes </th>
   <th style="text-align:left;"> copd </th>
   <th style="text-align:left;"> asthma </th>
   <th style="text-align:left;"> immunosuppression </th>
   <th style="text-align:left;"> hypertension </th>
   <th style="text-align:left;"> other_diseases </th>
   <th style="text-align:left;"> cardiovascular </th>
   <th style="text-align:left;"> obesity </th>
   <th style="text-align:left;"> chronic_kidney_failure </th>
   <th style="text-align:left;"> smoker </th>
   <th style="text-align:left;"> another_case </th>
   <th style="text-align:left;"> outcome </th>
   <th style="text-align:left;"> icu </th>
   <th style="text-align:left;"> y </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 42 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> live </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 51 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> live </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 51 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> live </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 57 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> die </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 44 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> live </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:left;"> 40 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> NA </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:left;"> live </td>
  </tr>
</tbody>
</table>

`````

:::
:::
