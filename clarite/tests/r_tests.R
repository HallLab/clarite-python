# This script loads several datasets from the survey library, tests the R version of CLARITE, and saves results to use in pytest
# Note that most examples run a single glm and compare it to the concatenated results of several EWAS runs (one per variable)
# in order to test all variables run at the same time (covariate values are ignored by clarite).

library(devtools)
install.packages('survey')
install_github('HallLab/clarite')
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

compare_ewas <- function(header, glm_result, ewas_result){
  # For each variable in the glm_result data.frame, Compare results to the ewas_result data.frame
  print(header)
  print("-----------------")
  variable_list <- glm_result$Variable
  for (variable in variable_list){
    print(paste("  ", variable, sep=""))
    # Compare Beta
    glm_beta <- glm_result[glm_result$Variable==variable, 'Beta']
    ewas_beta <- ewas_result[ewas_result$Variable==variable, 'Beta']
    match_beta <- isTRUE(all.equal(glm_beta, ewas_beta))
    print(paste("    ", " Beta: survey (", glm_beta, ") == clarite (", ewas_beta, ")?  ", match_beta, sep=""))
    # Compare Std Error
    glm_se <- glm_result[glm_result$Variable==variable, 'SE']
    ewas_se <- ewas_result[ewas_result$Variable==variable, 'SE']
    match_se <- isTRUE(all.equal(glm_se, ewas_se))
    print(paste("    ", " SE: survey (", glm_se, ") == clarite (", ewas_se, ")?  ", match_se, sep=""))
    # Compare Diff_AIC
    glm_aic <- glm_result[glm_result$Variable==variable, 'Diff_AIC']
    ewas_aic <- ewas_result[ewas_result$Variable==variable, 'Diff_AIC']
    match_aic <- isTRUE(all.equal(glm_aic, ewas_aic))
    print(paste("    ", " Diff_AIC: survey (", glm_aic, ") == clarite (", ewas_aic, ")?  ", match_aic, sep=""))
    # Compare Pval
    glm_pval <- glm_result[glm_result$Variable==variable, 'pval']
    ewas_pval <- ewas_result[ewas_result$Variable==variable, 'pval']
    match_pval <- isTRUE(all.equal(glm_pval, ewas_pval, tolerance = 1e-4))
    print(paste("    ", " pval: survey (", glm_pval, ") == clarite (", ewas_pval, ")?  ", match_pval, sep=""))
    
    if (!match_beta || !match_se || !match_pval){
      stop(paste("Discrepancy for ", variable, sep=""))
    }
  }
}

get_glm_result <- function(variable, glm_full, glm_restricted=NULL, use_weights=TRUE) {
  # Gathers results from the glm for continuous variables
  # Calculates results from two glms for categorical/binary variables
  if(is.null(glm_restricted)){
    # No restricted model = continuous variable
    glm_table <- summary(glm_full)$coefficients
    beta <- glm_table[variable, 'Estimate']
    se <- glm_table[variable, 'Std. Error']
    pval <- glm_table[variable, 'Pr(>|t|)']
    return(data.frame("Variable"=variable,
                      "Beta"=beta,
                      "SE"=se,
                      "Diff_AIC"=NA,
                      "pval"=pval))
  } else {
    # Restricted model exists = categorical variable (differences between stats::anova and survey::anova)
    if(use_weights){
      lrt <-  anova(glm_full, glm_restricted, method = "LRT")
      pval <- lrt$p
    } else {
      lrt <-  anova(glm_full, glm_restricted, test="LRT")
      pval <- lrt$`Pr(>Chi)`[2]
    }
    diff_aic <- glm_full$aic - glm_restricted$aic
    return(data.frame("Variable"=variable,
                      "Beta"=NA,
                      "SE"=NA,
                      "Diff_AIC"=diff_aic,
                      "pval"=pval))
  }
}

# Change to the output folder
current_dir <- getwd()
output_dir <- file.path(current_dir, "r_test_output")
dir.create(output_dir)
setwd(output_dir)

#################
# fpc Test data #
#################
# One outcome (y) and one variable (x)
data(fpc)
fpc <- add_id_col(fpc)
# Add an outcome variable
fpc$y <- fpc$x + (fpc$stratid * 2) + (fpc$psuid * 0.5) 
write.csv(fpc, 'fpc_data.csv', row.names=FALSE)

# No weights
glm_result_fpcnoweights <- rbind(get_glm_result("x", glm(y~x, data=fpc), use_weights=FALSE))
ewas_result_fpcnoweights <- ewas(d=fpc, cont_vars="x", cat_vars=NULL, y="y", regression_family="gaussian", min_n=1)
write.csv(ewas_result_fpcnoweights, 'fpc_noweights_result.csv', row.names=FALSE)
compare_ewas("fpc: No Weights", glm_result_fpcnoweights, ewas_result_fpcnoweights)

# Test without specifying fpc
withoutfpc <- svydesign(weights=~weight, ids=~psuid, strata=~stratid, data=fpc, nest=TRUE)
glm_result_withoutfpc <- rbind(get_glm_result("x", svyglm(y~x, design=withoutfpc)))
ewas_result_withoutfpc <- ewas(d=fpc, cont_vars = "x", cat_vars = NULL,
                               y="y", regression_family="gaussian",
                               weights="weight", ids="psuid", strata="stratid", nest=TRUE, min_n = 1)
write.csv(ewas_result_withoutfpc, 'fpc_withoutfpc_result.csv', row.names=FALSE)
compare_ewas("fpc: Without FPC", glm_result_withoutfpc, ewas_result_withoutfpc)

# Test with specifying fpc
withfpc <- svydesign(weights=~weight, ids=~psuid, strata=~stratid, fpc=~Nh, data=fpc, nest=TRUE)
glm_result_withfpc <- rbind(get_glm_result("x", svyglm(y~x, design=withfpc)))
ewas_result_withfpc <- ewas(d=fpc, cont_vars = "x", cat_vars = NULL,
                            y="y", regression_family="gaussian",
                            weights="weight", ids="psuid", strata="stratid", fpc="Nh", nest=TRUE, min_n = 1)
write.csv(ewas_result_withfpc, 'fpc_withfpc_result.csv', row.names=FALSE)
compare_ewas("fpc: With FPC", glm_result_withfpc, ewas_result_withfpc)

# Test with specifying fpc and no strata
# Have to make fpc identical now that it is one strata
fpc_nostrat <- fpc
fpc_nostrat$Nh <- 30
write.csv(fpc_nostrat, 'fpc_nostrat_data.csv', row.names=FALSE)
withfpc_nostrata <- svydesign(weights=~weight, ids=~psuid, strata=NULL, fpc=~Nh, data=fpc_nostrat)
glm_result_withfpc_nostrata <- rbind(get_glm_result("x", svyglm(y~x, design=withfpc_nostrata)))
ewas_result_withfpc_nostrata <- ewas(d=fpc_nostrat, cont_vars = "x", cat_vars = NULL,
                                     y="y", regression_family="gaussian",
                                     weights="weight", ids="psuid", strata=NULL, fpc="Nh", min_n = 1)
write.csv(ewas_result_withfpc_nostrata, 'fpc_withfpc_nostrat_result.csv', row.names=FALSE)
compare_ewas("fpc: With FPC, No Strata", glm_result_withfpc_nostrata, ewas_result_withfpc_nostrata)

#################
# api Test data #
#################
# one outcome (api00) and 3 continuous variables (ell, meals, mobility)
data(api)
apipop <- add_id_col(apipop)
write.csv(apipop, 'apipop_data.csv', row.names=FALSE)
apistrat <- add_id_col(apistrat)
write.csv(apistrat, 'apistrat_data.csv', row.names=FALSE)
apiclus1 <- add_id_col(apiclus1)
write.csv(apiclus1, 'apiclus1_data.csv', row.names=FALSE)

# Full population no weights
glm_apipop <- glm(api00~ell+meals+mobility, data=apipop)
glm_result_apipop <- rbind(
  get_glm_result("ell", glm_apipop, use_weights = FALSE),
  get_glm_result("meals", glm_apipop, use_weights = FALSE),
  get_glm_result("mobility", glm_apipop, use_weights = FALSE)
)
ewas_result_apipop <- rbind(
  ewas(d=apipop, cont_vars="ell", cat_vars=NULL,
       cont_covars = c("meals", "mobility"), cat_covars = NULL,
       y="api00", regression_family="gaussian", min_n=1),
  ewas(d=apipop, cont_vars="meals", cat_vars=NULL,
       cont_covars = c("ell", "mobility"), cat_covars = NULL,
       y="api00", regression_family="gaussian", min_n=1),
  ewas(d=apipop, cont_vars="mobility", cat_vars=NULL,
       cont_covars = c("ell", "meals"), cat_covars = NULL,
       y="api00", regression_family="gaussian", min_n=1)
)
write.csv(ewas_result_apipop, 'api_apipop_result.csv', row.names=FALSE)
compare_ewas("api: apipop", glm_result_apipop, ewas_result_apipop)

# stratified sample (no clusters) with fpc
dstrat <- svydesign(id=~1, strata=~stype, weights=~pw, data=apistrat, fpc=~fpc)
glm_apistrat <- svyglm(api00~ell+meals+mobility, design=dstrat)
glm_result_apistrat <- rbind(
  get_glm_result("ell", glm_apistrat),
  get_glm_result("meals", glm_apistrat),
  get_glm_result("mobility", glm_apistrat)
)
ewas_result_apistrat <- rbind(
  ewas(d=apistrat, cont_vars="ell", cat_vars=NULL,
       cont_covars = c("meals", "mobility"), cat_covars = NULL,
       y="api00", regression_family="gaussian", min_n=1,
       weights="pw", ids=NULL, strata="stype", fpc="fpc"),
  ewas(d=apistrat, cont_vars="meals", cat_vars=NULL,
       cont_covars = c("ell", "mobility"), cat_covars = NULL,
       y="api00", regression_family="gaussian", min_n=1,
       weights="pw", ids=NULL, strata="stype", fpc="fpc"),
  ewas(d=apistrat, cont_vars="mobility", cat_vars=NULL,
       cont_covars = c("ell", "meals"), cat_covars = NULL,
       y="api00", regression_family="gaussian", min_n=1,
       weights="pw", ids=NULL, strata="stype", fpc="fpc")
)
write.csv(ewas_result_apistrat, 'api_apistrat_result.csv', row.names=FALSE)
compare_ewas("api: apistrat", glm_result_apistrat, ewas_result_apistrat)

# one-stage cluster sample (no strata) with fpc
dclus1 <- svydesign(id=~dnum, weights=~pw, data=apiclus1, fpc=~fpc)
glm_apiclus1 <- svyglm(api00~ell+meals+mobility, design=dclus1)
glm_result_apiclus1 <- rbind(
  get_glm_result("ell", glm_apiclus1),
  get_glm_result("meals", glm_apiclus1),
  get_glm_result("mobility", glm_apiclus1)
)
ewas_result_apiclus1 <- rbind(
  ewas(d=apiclus1, cont_vars="ell", cat_vars=NULL,
       cont_covars = c("meals", "mobility"), cat_covars = NULL,
       y="api00", regression_family="gaussian", min_n=1,
       weights="pw", ids="dnum", strata=NULL, fpc="fpc"),
  ewas(d=apiclus1, cont_vars="meals", cat_vars=NULL,
       cont_covars = c("ell", "mobility"), cat_covars = NULL,
       y="api00", regression_family="gaussian", min_n=1,
       weights="pw", ids="dnum", strata=NULL, fpc="fpc"),
  ewas(d=apiclus1, cont_vars="mobility", cat_vars=NULL,
       cont_covars = c("ell", "meals"), cat_covars = NULL,
       y="api00", regression_family="gaussian", min_n=1,
       weights="pw", ids="dnum", strata=NULL, fpc="fpc")
)
write.csv(ewas_result_apiclus1, 'api_apiclus1_result.csv', row.names=FALSE)
compare_ewas("api: apiclus1", glm_result_apiclus1, ewas_result_apiclus1)

####################
# NHANES Test data #
####################
# A data frame with 8591 observations on the following 7 variables.
# SDMVPSU - Primary sampling units
# SDMVSTRA - Sampling strata
# WTMEC2YR - Sampling weights
# HI_CHOL - Binary: 1 for total cholesterol over 240mg/dl, 0 under 240mg/dl
# race - Categorical (1=Hispanic, 2=non-Hispanic white, 3=non-Hispanic black, 4=other)
# agecat  - Categorical Age group(0,19] (19,39] (39,59] (59,Inf]
# RIAGENDR - Binary: Gender: 1=male, 2=female

data(nhanes)
nhanes <- add_id_col(nhanes)
# Update types (all previous tests were using continuous)
# Don't update binary outcome (HI_CHOL) since outcome must be continuous
nhanes$HI_CHOL <- as.factor(nhanes$HI_CHOL)
nhanes$race <- as.factor(nhanes$race)
nhanes$agecat <- as.factor(nhanes$agecat)
nhanes$RIAGENDR <- as.factor(nhanes$RIAGENDR)
write.csv(nhanes, 'nhanes_data.csv', row.names=FALSE)

# Full population no weights
glm_nhanes_noweights <- glm(HI_CHOL~race+agecat+RIAGENDR, family=binomial(link="logit"), data=nhanes)
glm_result_nhanes_noweights <- rbind(
  get_glm_result("race", glm_nhanes_noweights, glm(HI_CHOL~agecat+RIAGENDR, family=binomial(link="logit"), data=nhanes), use_weights=FALSE),
  get_glm_result("agecat", glm_nhanes_noweights, glm(HI_CHOL~race+RIAGENDR, family=binomial(link="logit"), data=nhanes), use_weights=FALSE),
  get_glm_result("RIAGENDR", glm_nhanes_noweights, glm(HI_CHOL~race+agecat, family=binomial(link="logit"), data=nhanes), use_weights=FALSE)
)
ewas_result_nhanes_noweights <- rbind(
  ewas(d=nhanes, cont_vars=NULL, cat_vars="race",
       cont_covars = NULL, cat_covars = c("agecat", "RIAGENDR"),
       y="HI_CHOL", regression_family="binomial", min_n=1),
  ewas(d=nhanes, cont_vars=NULL, cat_vars="agecat",
       cont_covars = NULL, cat_covars = c("race", "RIAGENDR"),
       y="HI_CHOL", regression_family="binomial", min_n=1),
  ewas(d=nhanes, cont_vars=NULL, cat_vars="RIAGENDR",
       cont_covars = NULL, cat_covars = c("race", "agecat"),
       y="HI_CHOL", regression_family="binomial", min_n=1)
)
write.csv(ewas_result_nhanes_noweights, 'nhanes_noweights_result.csv', row.names=FALSE)
compare_ewas("nhanes: noweights", glm_result_nhanes_noweights, ewas_result_nhanes_noweights)


# Full design: cluster, strata, weights
dnhanes_complete <- svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes)
glm_nhanes_complete <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=dnhanes_complete, family=binomial(link="logit"))
glm_result_nhanes_complete <- rbind(
  get_glm_result("race", glm_nhanes_complete, svyglm(HI_CHOL~agecat+RIAGENDR, design=dnhanes_complete, family=binomial(link="logit"))),
  get_glm_result("agecat", glm_nhanes_complete, svyglm(HI_CHOL~race+RIAGENDR, design=dnhanes_complete, family=binomial(link="logit"))),
  get_glm_result("RIAGENDR", glm_nhanes_complete, svyglm(HI_CHOL~race+agecat, design=dnhanes_complete, family=binomial(link="logit")))
)
ewas_result_nhanes_complete <- rbind(
  ewas(d=nhanes, cont_vars=NULL, cat_vars="race",
       cont_covars = NULL, cat_covars = c("agecat", "RIAGENDR"),
       y="HI_CHOL", regression_family="binomial", min_n=1,
       weights="WTMEC2YR", ids="SDMVPSU", strata="SDMVSTRA", fpc=NULL, nest=TRUE),
  ewas(d=nhanes, cont_vars=NULL, cat_vars="agecat",
       cont_covars = NULL, cat_covars = c("race", "RIAGENDR"),
       y="HI_CHOL", regression_family="binomial", min_n=1,
       weights="WTMEC2YR", ids="SDMVPSU", strata="SDMVSTRA", fpc=NULL, nest=TRUE),
  ewas(d=nhanes, cont_vars=NULL, cat_vars="RIAGENDR",
       cont_covars = NULL, cat_covars = c("race", "agecat"),
       y="HI_CHOL", regression_family="binomial", min_n=1,
       weights="WTMEC2YR", ids="SDMVPSU", strata="SDMVSTRA", fpc=NULL, nest=TRUE)
)
write.csv(ewas_result_nhanes_complete, 'nhanes_complete_result.csv', row.names=FALSE)
compare_ewas("nhanes: complete", glm_result_nhanes_complete, ewas_result_nhanes_complete)

# Weights Only
dnhanes_weightsonly <- svydesign(id=~1, weights=~WTMEC2YR, data=nhanes)
glm_nhanes_weightsonly <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=dnhanes_weightsonly, family=binomial(link="logit"))
glm_result_nhanes_weightsonly <- rbind(
  get_glm_result("race", glm_nhanes_weightsonly, svyglm(HI_CHOL~agecat+RIAGENDR, design=dnhanes_weightsonly, family=binomial(link="logit"))),
  get_glm_result("agecat", glm_nhanes_weightsonly, svyglm(HI_CHOL~race+RIAGENDR, design=dnhanes_weightsonly, family=binomial(link="logit"))),
  get_glm_result("RIAGENDR", glm_nhanes_weightsonly, svyglm(HI_CHOL~race+agecat, design=dnhanes_weightsonly, family=binomial(link="logit")))
)
ewas_result_nhanes_weightsonly <- rbind(
  ewas(d=nhanes, cont_vars=NULL, cat_vars="race",
       cont_covars = NULL, cat_covars = c("agecat", "RIAGENDR"),
       y="HI_CHOL", regression_family="binomial", min_n=1,
       weights="WTMEC2YR"),
  ewas(d=nhanes, cont_vars=NULL, cat_vars="agecat",
       cont_covars = NULL, cat_covars = c("race", "RIAGENDR"),
       y="HI_CHOL", regression_family="binomial", min_n=1,
       weights="WTMEC2YR"),
  ewas(d=nhanes, cont_vars=NULL, cat_vars="RIAGENDR",
       cont_covars = NULL, cat_covars = c("race", "agecat"),
       y="HI_CHOL", regression_family="binomial", min_n=1,
       weights="WTMEC2YR")
)
write.csv(ewas_result_nhanes_weightsonly, 'nhanes_weightsonly_result.csv', row.names=FALSE)
compare_ewas("nhanes: weights only", glm_result_nhanes_weightsonly, ewas_result_nhanes_weightsonly)

#################
# NHANES Lonely #
#################
# Lonely PSU (only one PSU in a stratum)
nhanes_lonely <- nhanes
# Make Lonely PSUs by dropping some rows
nhanes_lonely <- nhanes_lonely[!((nhanes_lonely$SDMVSTRA==81) & (nhanes_lonely$SDMVPSU!=1)),]
nhanes_lonely <- nhanes_lonely[!((nhanes_lonely$SDMVSTRA==82) & (nhanes_lonely$SDMVPSU!=1)),]
nhanes_lonely <- nhanes_lonely[!((nhanes_lonely$SDMVSTRA==83) & (nhanes_lonely$SDMVPSU!=1)),]
nhanes_lonely <- nhanes_lonely[!((nhanes_lonely$SDMVSTRA==84) & (nhanes_lonely$SDMVPSU!=1)),]
nhanes_lonely <- nhanes_lonely[!((nhanes_lonely$SDMVSTRA==85) & (nhanes_lonely$SDMVPSU!=1)),]
nhanes_lonely <- nhanes_lonely[!((nhanes_lonely$SDMVSTRA==86) & (nhanes_lonely$SDMVPSU!=1)),]
nhanes_lonely <- nhanes_lonely[!((nhanes_lonely$SDMVSTRA==87) & (nhanes_lonely$SDMVPSU!=1)),]
print(paste("Removed", nrow(nhanes) - nrow(nhanes_lonely), "rows to make lonely PSUs", sep=" "))
# Save data
write.csv(nhanes_lonely, 'nhanes_lonely_data.csv', row.names=FALSE)

get_lonely_glm_results <- function(setting){
  options(survey.lonely.psu=setting)
  dnhanes_lonely <- svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely)
  glm_nhanes_lonely <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely, family=binomial(link="logit")))
  glm_result_nhanes_lonely <- rbind(
    get_glm_result("race", glm_nhanes_lonely, svyglm(HI_CHOL~agecat+RIAGENDR, design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely, family=binomial(link="logit")))),
    get_glm_result("agecat", glm_nhanes_lonely, svyglm(HI_CHOL~race+RIAGENDR, design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely, family=binomial(link="logit")))),
    get_glm_result("RIAGENDR", glm_nhanes_lonely, svyglm(HI_CHOL~race+agecat, design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely, family=binomial(link="logit"))))
  )
  return(glm_result_nhanes_lonely)
}
get_lonely_ewas_results <- function(setting){
  options(survey.lonely.psu=setting)
  ewas_result_nhanes_lonely <- rbind(
    ewas(d=nhanes_lonely, cont_vars=NULL, cat_vars="race",
         cont_covars = NULL, cat_covars = c("agecat", "RIAGENDR"),
         y="HI_CHOL", regression_family="binomial", min_n=1,
         weights="WTMEC2YR", ids="SDMVPSU", strata="SDMVSTRA", fpc=NULL, nest=TRUE),
    ewas(d=nhanes_lonely, cont_vars=NULL, cat_vars="agecat",
         cont_covars = NULL, cat_covars = c("race", "RIAGENDR"),
         y="HI_CHOL", regression_family="binomial", min_n=1,
         weights="WTMEC2YR", ids="SDMVPSU", strata="SDMVSTRA", fpc=NULL, nest=TRUE),
    ewas(d=nhanes_lonely, cont_vars=NULL, cat_vars="RIAGENDR",
         cont_covars = NULL, cat_covars = c("race", "agecat"),
         y="HI_CHOL", regression_family="binomial", min_n=1,
         weights="WTMEC2YR", ids="SDMVPSU", strata="SDMVSTRA", fpc=NULL, nest=TRUE)
  )
  return(ewas_result_nhanes_lonely)
}
# Certainty
glm_result_nhanes_certainty <- get_lonely_glm_results("certainty")
ewas_result_nhanes_certainty <- get_lonely_ewas_results("certainty")
write.csv(ewas_result_nhanes_certainty, 'nhanes_certainty_result.csv', row.names=FALSE)
compare_ewas("nhanes lonely: certainty", glm_result_nhanes_certainty, ewas_result_nhanes_certainty)

# Adjust
glm_result_nhanes_adjust <- get_lonely_glm_results("adjust")
ewas_result_nhanes_adjust <- get_lonely_ewas_results("adjust")
write.csv(ewas_result_nhanes_adjust, 'nhanes_adjust_result.csv', row.names=FALSE)
compare_ewas("nhanes lonely: adjust", glm_result_nhanes_adjust, ewas_result_nhanes_adjust)
