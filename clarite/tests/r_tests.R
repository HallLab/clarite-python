# This script loads several datasets from the survey library, tests the R version of CLARITE, and saves results to use in pytest

library(devtools)
install.packages('survey')
#install_github('HallLab/clarite')
install.packages("C:/Users/jrm5100/Documents/Code/clarite", repos=NULL, type="source")
library('survey')
library('clarite')

####################
# Useful Functions #
####################

add_id_col <- function(df){
  df$ID <- 1:nrow(df)
  df <- df[,c("ID",setdiff(names(df),"ID"))]
  df
}

compare_ewas <- function(glm_result, ewas_result, variable){
  glm_table <- summary(glm_result)$coefficients
  # Compare Beta
  glm_beta <- glm_table[variable, 'Estimate']
  ewas_beta <- ewas_result[ewas_result$Variable==variable, 'Beta']
  if(!(glm_beta == ewas_beta)){
    stop(paste("Variable", variable, "Beta:", glm_beta, "in glm vs", ewas_beta, "in ewas", sep=" "))
  }
  # Compare Std Error
  glm_se <- glm_table[variable, 'Std. Error']
  ewas_se <- ewas_result[ewas_result$Variable==variable, 'SE']
  if(!(glm_se == ewas_se)){
    stop(paste("Variable", variable, "SE:", glm_se, "in glm vs", ewas_se, "in ewas", sep=" "))
  }
  # Compare Pval
  glm_pval <- glm_table[variable, 'Pr(>|t|)']
  ewas_pval <- ewas_result[ewas_result$Variable==variable, 'pval']
  if(!(glm_pval == ewas_pval)){
    stop(paste("Variable", variable, "pval:", glm_pval, "in glm vs", ewas_pval, "in ewas", sep=" "))
  }
  print(paste("    ", variable, " Matches", sep=""))
}

# Change to the output folder
current_dir <- getwd()
output_dir <- file.path(current_dir, "r_test_output")
dir.create(output_dir)
setwd(output_dir)

#################
# fpc Test data #
#################
data(fpc)
fpc <- add_id_col(fpc)
# Add an outcome variable
fpc$y <- fpc$x + (fpc$stratid * 2) + (fpc$psuid * 0.5) 
write.csv(fpc, 'fpc_data.csv')

# Test without specifying fpc
withoutfpc <- svydesign(weights=~weight, ids=~psuid, strata=~stratid, data=fpc, nest=TRUE)
glm_result_withoutfpc <- svyglm(y~x, design=withoutfpc)
ewas_result_withoutfpc <- ewas(d=fpc, cont_vars = "x", cat_vars = NULL,
                               y="y", regression_family="gaussian",
                               weights="weight", ids="psuid", strata="stratid", nest=TRUE, min_n = 1)
write.csv(ewas_result_withoutfpc, 'fpc_withoutfpc_result.csv')
print("fpc: Without FPC")
compare_ewas(glm_result_withoutfpc, ewas_result_withoutfpc, "x")

# Test with specifying fpc
withfpc <- svydesign(weights=~weight, ids=~psuid, strata=~stratid, fpc=~Nh, data=fpc, nest=TRUE)
glm_result_withfpc <- svyglm(y~x, design=withfpc)
ewas_result_withfpc <- ewas(d=fpc, cont_vars = "x", cat_vars = NULL,
                            y="y", regression_family="gaussian",
                            weights="weight", ids="psuid", strata="stratid", fpc="Nh", nest=TRUE, min_n = 1)
write.csv(ewas_result_withfpc, 'fpc_withfpc_result.csv')
print("fpc: With FPC")
compare_ewas(glm_result_withfpc, ewas_result_withfpc, "x")

#################
# api Test data #
#################
data(api)
apistrat <- add_id_col(apistrat)
write.csv(apistrat, 'apistrat_data.csv')
apiclus1 <- add_id_col(apiclus1)
write.csv(apiclus1, 'apiclus1_data.csv')

# stratified sample (no clusters) with fpc
dstrat <- svydesign(id=~1, strata=~stype, weights=~pw, data=apistrat, fpc=~fpc)
glm_result_apistrat <- svyglm(api00~ell+meals+mobility, design=dstrat)
ewas_result_apistrat <- ewas(d=apistrat, cont_vars = "ell", cat_vars = NULL,
                             cont_covars = c("meals", "mobility"), cat_covars = NULL,
                             y="api00", regression_family="gaussian",
                             weights="pw", ids=NULL, strata="stype", fpc="fpc", min_n = 1)
print("api: apistrat for ell")
compare_ewas(glm_result_apistrat, ewas_result_apistrat, "ell")
write.csv(ewas_result_apistrat, 'api_dstrat_result.csv')

# one-stage cluster sample (no strata) with fpc
dclus1 <- svydesign(id=~dnum, weights=~pw, data=apiclus1, fpc=~fpc)
glm_result_apiclus1 <- svyglm(api00~ell+meals+mobility, design=dclus1)
ewas_result_apiclus1 <- ewas(d=apiclus1, cont_vars = "ell", cat_vars = NULL,
                             cont_covars = c("meals", "mobility"), cat_covars = NULL,
                             y="api00", regression_family="gaussian",
                             weights="pw", ids="dnum", fpc="fpc", min_n = 1)
print("api: apiclus1 for ell")
compare_ewas(glm_result_apiclus1, ewas_result_apiclus1, "ell")
write.csv(ewas_result_apiclus1, 'api_apiclus1_result.csv')

####################
# nhanes Test data #
####################
# A data frame with 8591 observations on the following 7 variables.
# SDMVPSU - Primary sampling units
# SDMVSTRA - Sampling strata
# WTMEC2YR - Sampling weights
# HI_CHOL - Numeric vector: 1 for total cholesterol over 240mg/dl, 0 under 240mg/dl
# race - Categorical (1=Hispanic, 2=non-Hispanic white, 3=non-Hispanic black, 4=other)
# agecat  - Categorical Age group(0,19] (19,39] (39,59] (59,Inf]
# RIAGENDR - Binary: Gender: 1=male, 2=female

data(nhanes)
write.csv(nhanes, 'nhanes_data.csv')

# Cluster, Strata, Weights
design <- svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes)
