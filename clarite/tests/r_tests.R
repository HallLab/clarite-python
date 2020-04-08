# This script loads several datasets from the survey library abnd runs GLMs to compare results in CLARITE

library(devtools)
if (!require('survey')) install.packages('survey', repos = "http://cran.us.r-project.org"); library('survey')
#install_github('HallLab/clarite')
#library('clarite')

####################
# Useful Functions #
####################

get_glm_result <- function(variable, glm_full, glm_restricted=NULL, use_weights=TRUE, alt_name=NULL) {
  # Gathers results from the glm for continuous variables
  # Calculates results from two glms for categorical/binary variables
  if(is.null(glm_restricted)){
    # No restricted model = continuous variable
    # If a binary variable is being tested like python, the indexed name will be different and specified by alt_name
    if(is.null(alt_name)){
      var_idx_name <- variable
    } else {
      var_idx_name <- alt_name
    }
    glm_table <- summary(glm_full)$coefficients
    beta <- glm_table[var_idx_name, 'Estimate']
    se <- glm_table[var_idx_name, 'Std. Error']
    # Pval is based on a t-value for gaussian and a z-value for binomial
    if ('Pr(>|t|)' %in% colnames(glm_table)){
      pval <- glm_table[var_idx_name, 'Pr(>|t|)']
    } else {
      pval <- glm_table[var_idx_name, 'Pr(>|z|)']
    }
    return(data.frame("Variable"=variable,
                      "Beta"=beta,
                      "SE"=se,
                      "Diff_AIC"=NA,
                      "pvalue"=pval))
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
                      "pvalue"=pval))
  }
}

update_binary_result <- function(ewas_result, new_bin_result){
  # Change results for binary variables so that a LRT is not performed, in order to match the python version
  ewas_result$Beta[match(new_bin_result$Variable, ewas_result$Variable)] <- new_bin_result$Beta
  ewas_result$SE[match(new_bin_result$Variable, ewas_result$Variable)] <- new_bin_result$SE
  ewas_result$Diff_AIC[match(new_bin_result$Variable, ewas_result$Variable)] <- new_bin_result$Diff_AIC
  ewas_result$pvalue[match(new_bin_result$Variable, ewas_result$Variable)] <- new_bin_result$pvalue
  return(ewas_result)
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
# Add an outcome variable
fpc$y <- fpc$x + (fpc$stratid * 2) + (fpc$psuid * 0.5) 
write.csv(fpc, 'fpc_data.csv', row.names=FALSE)
# Reload to ensure there are no rounding differences
fpc <- read.csv(file = 'fpc_data.csv')

# No weights
glm_result_fpcnoweights <- rbind(get_glm_result("x", glm(y~x, data=fpc), use_weights=FALSE))
# No binary variables- python should match
write.csv(glm_result_fpcnoweights, 'fpc_noweights_result.csv', row.names=FALSE)

# Test without specifying fpc
withoutfpc <- svydesign(weights=~weight, ids=~psuid, strata=~stratid, data=fpc, nest=TRUE)
glm_result_withoutfpc <- rbind(get_glm_result("x", svyglm(y~x, design=withoutfpc)))
# No binary variables- python should match
write.csv(glm_result_withoutfpc, 'fpc_withoutfpc_result.csv', row.names=FALSE)

# Test with specifying fpc
withfpc <- svydesign(weights=~weight, ids=~psuid, strata=~stratid, fpc=~Nh, data=fpc, nest=TRUE)
glm_result_withfpc <- rbind(get_glm_result("x", svyglm(y~x, design=withfpc)))
# No binary variables- python should match
write.csv(glm_result_withfpc, 'fpc_withfpc_result.csv', row.names=FALSE)

# Test with specifying fpc and no strata
# Have to make fpc identical now that it is one strata
fpc_nostrat <- fpc
fpc_nostrat$Nh <- 30
write.csv(fpc_nostrat, 'fpc_nostrat_data.csv', row.names=FALSE)
withfpc_nostrata <- svydesign(weights=~weight, ids=~psuid, strata=NULL, fpc=~Nh, data=fpc_nostrat)
glm_result_withfpc_nostrata <- rbind(get_glm_result("x", svyglm(y~x, design=withfpc_nostrata)))
# No binary variables- python should match
write.csv(glm_result_withfpc_nostrata, 'fpc_withfpc_nostrat_result.csv', row.names=FALSE)

#################
# api Test data #
#################
# one outcome (api00) and 3 continuous variables (ell, meals, mobility)
data(api)
write.csv(apipop, 'apipop_data.csv', row.names=FALSE)
apipop <- read.csv(file = 'apipop_data.csv')

write.csv(apistrat, 'apistrat_data.csv', row.names=FALSE)
apistrat <- read.csv(file = 'apistrat_data.csv')

write.csv(apiclus1, 'apiclus1_data.csv', row.names=FALSE)
apiclus1 <- read.csv(file = 'apiclus1_data.csv')

# Full population no weights
glm_apipop <- glm(api00~ell+meals+mobility, data=apipop)
glm_result_apipop <- rbind(
  get_glm_result("ell", glm_apipop, use_weights = FALSE),
  get_glm_result("meals", glm_apipop, use_weights = FALSE),
  get_glm_result("mobility", glm_apipop, use_weights = FALSE)
)
# No binary variables- python should match
write.csv(glm_result_apipop, 'api_apipop_result.csv', row.names=FALSE)

# stratified sample (no clusters) with fpc
dstrat <- svydesign(id=~1, strata=~stype, weights=~pw, data=apistrat, fpc=~fpc)
glm_apistrat <- svyglm(api00~ell+meals+mobility, design=dstrat)
glm_result_apistrat <- rbind(
  get_glm_result("ell", glm_apistrat),
  get_glm_result("meals", glm_apistrat),
  get_glm_result("mobility", glm_apistrat)
)
# No binary variables- python should match
write.csv(glm_result_apistrat, 'api_apistrat_result.csv', row.names=FALSE)

# one-stage cluster sample (no strata) with fpc
dclus1 <- svydesign(id=~dnum, weights=~pw, data=apiclus1, fpc=~fpc)
glm_apiclus1 <- svyglm(api00~ell+meals+mobility, design=dclus1)
glm_result_apiclus1 <- rbind(
  get_glm_result("ell", glm_apiclus1),
  get_glm_result("meals", glm_apiclus1),
  get_glm_result("mobility", glm_apiclus1)
)
# No binary variables- python should match
write.csv(glm_result_apiclus1, 'api_apiclus1_result.csv', row.names=FALSE)

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
# Update types (all previous tests were using continuous)
# Don't update binary outcome (HI_CHOL) since outcome must be continuous
nhanes$HI_CHOL <- as.factor(nhanes$HI_CHOL)
nhanes$race <- as.factor(nhanes$race)
nhanes$agecat <- as.factor(nhanes$agecat)
nhanes$RIAGENDR <- as.factor(nhanes$RIAGENDR)
write.csv(nhanes, 'nhanes_data.csv', row.names=FALSE)
nhanes <- read.csv('nhanes_data.csv')
nhanes$HI_CHOL <- as.factor(nhanes$HI_CHOL)
nhanes$race <- as.factor(nhanes$race)
nhanes$agecat <- as.factor(nhanes$agecat)
nhanes$RIAGENDR <- as.factor(nhanes$RIAGENDR)

# Full population no weights
glm_nhanes_noweights <- glm(HI_CHOL~race+agecat+RIAGENDR, family=binomial(link="logit"), data=nhanes)
glm_result_nhanes_noweights <- rbind(
  get_glm_result("race", glm_nhanes_noweights, glm(HI_CHOL~agecat+RIAGENDR, family=binomial(link="logit"), data=nhanes), use_weights=FALSE),
  get_glm_result("agecat", glm_nhanes_noweights, glm(HI_CHOL~race+RIAGENDR, family=binomial(link="logit"), data=nhanes), use_weights=FALSE),
  get_glm_result("RIAGENDR", glm_nhanes_noweights, glm(HI_CHOL~race+agecat, family=binomial(link="logit"), data=nhanes), use_weights=FALSE)
)
write.csv(glm_result_nhanes_noweights, 'nhanes_noweights_result_r.csv', row.names=FALSE)
# RIAGENDR is binary, need to change calculation to match python
glm_result_nhanes_noweights <- update_binary_result(
  ewas_result = glm_result_nhanes_noweights,
  new_bin_result = get_glm_result("RIAGENDR", glm_nhanes_noweights, glm_restricted=NULL, use_weights=FALSE, alt_name="RIAGENDR2")
  )
write.csv(glm_result_nhanes_noweights, 'nhanes_noweights_result.csv', row.names=FALSE)


# Full design: cluster, strata, weights
dnhanes_complete <- svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes)
glm_nhanes_complete <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=dnhanes_complete, family=binomial(link="logit"))
print("**** Survey Design ****")
print(summary(dnhanes_complete))
print("***********************")
#print("HI_CHOL~race+agecat+RIAGENDR")
#print("HI_CHOL~agecat+RIAGENDR")
#print(summary(svyglm(HI_CHOL~race+agecat+RIAGENDR, design=dnhanes_complete, family=binomial(link="logit"))))
#print("===================")
#print(summary(svyglm(HI_CHOL~agecat+RIAGENDR, design=dnhanes_complete, family=binomial(link="logit"))))

glm_result_nhanes_complete <- rbind(
  get_glm_result("race", glm_nhanes_complete, svyglm(HI_CHOL~agecat+RIAGENDR, design=dnhanes_complete, family=binomial(link="logit"))),
  get_glm_result("agecat", glm_nhanes_complete, svyglm(HI_CHOL~race+RIAGENDR, design=dnhanes_complete, family=binomial(link="logit"))),
  get_glm_result("RIAGENDR", glm_nhanes_complete, svyglm(HI_CHOL~race+agecat, design=dnhanes_complete, family=binomial(link="logit")))
)
write.csv(glm_result_nhanes_complete, 'nhanes_complete_result_r.csv', row.names=FALSE)
# RIAGENDR is binary, need to change calculation to match python
glm_result_nhanes_complete <- update_binary_result(
  ewas_result = glm_result_nhanes_complete,
  new_bin_result = get_glm_result("RIAGENDR", glm_nhanes_complete, alt_name="RIAGENDR2")
  )
write.csv(glm_result_nhanes_complete, 'nhanes_complete_result.csv', row.names=FALSE)

# Weights Only
dnhanes_weightsonly <- svydesign(id=~1, weights=~WTMEC2YR, data=nhanes)
glm_nhanes_weightsonly <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=dnhanes_weightsonly, family=binomial(link="logit"))
glm_result_nhanes_weightsonly <- rbind(
  get_glm_result("race", glm_nhanes_weightsonly, svyglm(HI_CHOL~agecat+RIAGENDR, design=dnhanes_weightsonly, family=binomial(link="logit"))),
  get_glm_result("agecat", glm_nhanes_weightsonly, svyglm(HI_CHOL~race+RIAGENDR, design=dnhanes_weightsonly, family=binomial(link="logit"))),
  get_glm_result("RIAGENDR", glm_nhanes_weightsonly, svyglm(HI_CHOL~race+agecat, design=dnhanes_weightsonly, family=binomial(link="logit")))
)
write.csv(glm_result_nhanes_weightsonly, 'nhanes_weightsonly_result_r.csv', row.names=FALSE)
# RIAGENDR is binary, need to change calculation to match python
glm_result_nhanes_weightsonly <- update_binary_result(
  ewas_result = glm_result_nhanes_weightsonly,
  new_bin_result = get_glm_result("RIAGENDR", glm_nhanes_weightsonly, alt_name="RIAGENDR2")
  )
write.csv(glm_result_nhanes_weightsonly, 'nhanes_weightsonly_result.csv', row.names=FALSE)

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
# Don't need to reload nhanes_lonely since it is subset from a reload

get_lonely_glm_results <- function(setting, binary_as_continuous=FALSE){
  options(survey.lonely.psu=setting)
  glm_nhanes_lonely <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely), family=binomial(link="logit"))
  if(binary_as_continuous){
    glm_result_nhanes_lonely <- rbind(
      get_glm_result("race", glm_nhanes_lonely, svyglm(HI_CHOL~agecat+RIAGENDR, design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely), family=binomial(link="logit"))),
      get_glm_result("agecat", glm_nhanes_lonely, svyglm(HI_CHOL~race+RIAGENDR, design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely), family=binomial(link="logit"))),
      get_glm_result("RIAGENDR", glm_nhanes_lonely, alt_name="RIAGENDR2")
    )
  } else {
    glm_result_nhanes_lonely <- rbind(
      get_glm_result("race", glm_nhanes_lonely, svyglm(HI_CHOL~agecat+RIAGENDR, design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely), family=binomial(link="logit"))),
      get_glm_result("agecat", glm_nhanes_lonely, svyglm(HI_CHOL~race+RIAGENDR, design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely), family=binomial(link="logit"))),
      get_glm_result("RIAGENDR", glm_nhanes_lonely, svyglm(HI_CHOL~race+agecat, design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely), family=binomial(link="logit")))
    )
  }
  return(glm_result_nhanes_lonely)
}

# Certainty
glm_result_nhanes_certainty <- get_lonely_glm_results("remove", binary_as_continuous=FALSE)
write.csv(glm_result_nhanes_certainty, 'nhanes_certainty_result_r.csv', row.names=FALSE)
glm_result_nhanes_certainty <- get_lonely_glm_results("remove", binary_as_continuous=TRUE)
write.csv(glm_result_nhanes_certainty, 'nhanes_certainty_result.csv', row.names=FALSE)

# Adjust
glm_result_nhanes_adjust <- get_lonely_glm_results("adjust", binary_as_continuous=FALSE)
write.csv(glm_result_nhanes_adjust, 'nhanes_adjust_result_r.csv', row.names=FALSE)
glm_result_nhanes_adjust <- get_lonely_glm_results("adjust", binary_as_continuous=TRUE)
write.csv(glm_result_nhanes_adjust, 'nhanes_adjust_result.csv', row.names=FALSE)
