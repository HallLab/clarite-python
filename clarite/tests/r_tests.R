# This script loads several datasets from the survey library abnd runs GLMs to compare results in CLARITE

library(devtools)
if (!require('survey')) install.packages('survey', repos = "http://cran.us.r-project.org"); library('survey')

####################
# Useful Functions #
####################

write_result <- function(data, filename) {
  # Round numeric columns to limit precision
  if(is.numeric(data$Beta)){data$Beta <- formatC(data$Beta, format = "e", digits = 6)}
  if(is.numeric(data$SE)){data$SE <- formatC(as.numeric(data$SE), format = "e", digits = 6)}
  if(is.numeric(data$Diff_AIC)){data$Diff_AIC <- formatC(as.numeric(data$Diff_AIC), format = "e", digits = 6)}
  if(is.numeric(data$pvalue)){data$pvalue  <- formatC(as.numeric(data$pvalue), format = "e", digits = 6)}

  # Fix where "NA" was modified by the above
  data <- replace(data, data=="     NA", "NA")

  write.csv(data, filename, row.names=FALSE)
}

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
                      "N"=nobs(glm_full),
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
                      "N"=nobs(glm_full),
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
print("fpc Test data")
# One outcome (y) and one variable (x)
data(fpc)
# Add an outcome variable
fpc$y <- fpc$x + (fpc$stratid * 2) + (fpc$psuid * 0.5) 
write.csv(fpc, 'fpc_data.csv', row.names=FALSE)
# Reload to ensure there are no rounding differences
fpc <- read.csv(file = 'fpc_data.csv')

# No weights
glm_result_fpcnoweights <- rbind(get_glm_result("x", glm(y~x, data=fpc, na.action=na.omit), use_weights=FALSE))
# No binary variables- python should match
write_result(glm_result_fpcnoweights, 'fpc_noweights_result.csv')

# Test without specifying fpc
withoutfpc <- svydesign(weights=~weight, ids=~psuid, strata=~stratid, data=fpc, nest=TRUE)
glm_result_withoutfpc <- rbind(get_glm_result("x", svyglm(y~x, design=withoutfpc, na.action=na.omit)))
# No binary variables- python should match
write_result(glm_result_withoutfpc, 'fpc_withoutfpc_result.csv')

# Test with specifying fpc
withfpc <- svydesign(weights=~weight, ids=~psuid, strata=~stratid, fpc=~Nh, data=fpc, nest=TRUE)
glm_result_withfpc <- rbind(get_glm_result("x", svyglm(y~x, design=withfpc, na.action=na.omit)))
# No binary variables- python should match
write_result(glm_result_withfpc, 'fpc_withfpc_result.csv')

# Test with specifying fpc and no strata
# Have to make fpc identical now that it is one strata
fpc_nostrat <- fpc
fpc_nostrat$Nh <- 30
write.csv(fpc_nostrat, 'fpc_nostrat_data.csv', row.names=FALSE)
withfpc_nostrata <- svydesign(weights=~weight, ids=~psuid, strata=NULL, fpc=~Nh, data=fpc_nostrat)
glm_result_withfpc_nostrata <- rbind(get_glm_result("x", svyglm(y~x, design=withfpc_nostrata, na.action=na.omit)))
# No binary variables- python should match
write_result(glm_result_withfpc_nostrata, 'fpc_withfpc_nostrat_result.csv')

#################
# api Test data #
#################
print("api Test data")
# one outcome (api00) and 3 continuous variables (ell, meals, mobility)
data(api)
write.csv(apipop, 'apipop_data.csv', row.names=FALSE)
write.csv(apistrat, 'apistrat_data.csv', row.names=FALSE)
write.csv(apiclus1, 'apiclus1_data.csv', row.names=FALSE)
# Make a modified version
api_withna <- apipop
api_withna$api00[2:200] <- NA
write.csv(api_withna, 'apipop_withna_data.csv', row.names=FALSE)

apipop <- read.csv(file = 'apipop_data.csv')
apipop_withna <- read.csv(file = 'apipop_withna_data.csv')
apistrat <- read.csv(file = 'apistrat_data.csv')
apiclus1 <- read.csv(file = 'apiclus1_data.csv')

# Full population no weights
glm_apipop <- glm(api00~ell+meals+mobility, data=apipop, na.action=na.omit)
glm_result_apipop <- rbind(
  get_glm_result("ell", glm_apipop, use_weights = FALSE),
  get_glm_result("meals", glm_apipop, use_weights = FALSE),
  get_glm_result("mobility", glm_apipop, use_weights = FALSE)
)
# No binary variables- python should match
write_result(glm_result_apipop, 'api_apipop_result.csv')

# Full population no weights, with NA
glm_apipop_withna <- glm(api00~ell+meals+mobility, data=apipop_withna, na.action=na.omit)
glm_result_apipop_withna <- rbind(
  get_glm_result("ell", glm_apipop_withna, use_weights = FALSE),
  get_glm_result("meals", glm_apipop_withna, use_weights = FALSE),
  get_glm_result("mobility", glm_apipop_withna, use_weights = FALSE)
)
# No binary variables- python should match
write_result(glm_result_apipop_withna, 'api_apipop_withna_result.csv')

# stratified sample (no clusters) with fpc
dstrat <- svydesign(id=~1, strata=~stype, weights=~pw, data=apistrat, fpc=~fpc)
glm_apistrat <- svyglm(api00~ell+meals+mobility, design=dstrat, na.action=na.omit)
glm_result_apistrat <- rbind(
  get_glm_result("ell", glm_apistrat),
  get_glm_result("meals", glm_apistrat),
  get_glm_result("mobility", glm_apistrat)
)
# No binary variables- python should match
write_result(glm_result_apistrat, 'api_apistrat_result.csv')

# one-stage cluster sample (no strata) with fpc
dclus1 <- svydesign(id=~dnum, weights=~pw, data=apiclus1, fpc=~fpc)
glm_apiclus1 <- svyglm(api00~ell+meals+mobility, design=dclus1, na.action=na.omit)
glm_result_apiclus1 <- rbind(
  get_glm_result("ell", glm_apiclus1),
  get_glm_result("meals", glm_apiclus1),
  get_glm_result("mobility", glm_apiclus1)
)
# No binary variables- python should match
write_result(glm_result_apiclus1, 'api_apiclus1_result.csv')

####################
# NHANES Test data #
####################
print("NHANES Test data")
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
print("Full population no weights")
glm_nhanes_noweights <- glm(HI_CHOL~race+agecat+RIAGENDR, family=binomial(link="logit"), data=nhanes, na.action=na.omit)
glm_result_nhanes_noweights <- rbind(
  get_glm_result("race", glm_nhanes_noweights,
                 glm_restricted = glm(HI_CHOL~agecat+RIAGENDR, family=binomial(link="logit"), data=glm_nhanes_noweights$model),
                 use_weights=FALSE),
  get_glm_result("agecat", glm_nhanes_noweights,
                 glm_restricted = glm(HI_CHOL~race+RIAGENDR, family=binomial(link="logit"), data=glm_nhanes_noweights$model),
                 use_weights=FALSE),
  get_glm_result("RIAGENDR", glm_nhanes_noweights,
                 glm_restricted = glm(HI_CHOL~race+agecat, family=binomial(link="logit"), data=glm_nhanes_noweights$model),
                 use_weights=FALSE)
)
write_result(glm_result_nhanes_noweights, 'nhanes_noweights_result_r.csv')
# RIAGENDR is binary, need to change calculation to match python
glm_result_nhanes_noweights <- update_binary_result(
  ewas_result = glm_result_nhanes_noweights,
  new_bin_result = get_glm_result("RIAGENDR", glm_nhanes_noweights, glm_restricted=NULL, use_weights=FALSE, alt_name="RIAGENDR2")
  )
write_result(glm_result_nhanes_noweights, 'nhanes_noweights_result.csv')

# Full population no weights, with some categorical data missing
print("Full population no weights, with some categorical data missing")
nhanes_NAs <- nhanes
nhanes_NAs$race[2:800] <- NA
write.csv(nhanes_NAs, 'nhanes_NAs_data.csv', row.names=FALSE)
glm_nhanes_noweights <- glm(HI_CHOL~race+agecat+RIAGENDR, family=binomial(link="logit"), data=nhanes_NAs, na.action=na.omit)
glm_result_nhanes_noweights <- rbind(
  get_glm_result("race", glm_nhanes_noweights,
                 glm_restricted = glm(HI_CHOL~agecat+RIAGENDR, family=binomial(link="logit"), data=glm_nhanes_noweights$model),
                 use_weights=FALSE),
  get_glm_result("agecat", glm_nhanes_noweights,
                 glm_restricted = glm(HI_CHOL~race+RIAGENDR, family=binomial(link="logit"), data=glm_nhanes_noweights$model),
                 use_weights=FALSE),
  get_glm_result("RIAGENDR", glm_nhanes_noweights,
                 glm_restricted = glm(HI_CHOL~race+agecat, family=binomial(link="logit"), data=glm_nhanes_noweights$model),
                 use_weights=FALSE)
)
write_result(glm_result_nhanes_noweights, 'nhanes_noweights_withna_result_r.csv')
# RIAGENDR is binary, need to change calculation to match python
glm_result_nhanes_noweights <- update_binary_result(
  ewas_result = glm_result_nhanes_noweights,
  new_bin_result = get_glm_result("RIAGENDR", glm_nhanes_noweights, glm_restricted=NULL, use_weights=FALSE, alt_name="RIAGENDR2")
  )
write_result(glm_result_nhanes_noweights, 'nhanes_noweights_withna_result.csv')


# Full design: cluster, strata, weights
print("Full Design")
dnhanes_complete <- svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes)
glm_nhanes_complete <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=dnhanes_complete, family=binomial(link="logit"), na.action=na.omit)
glm_result_nhanes_complete <- rbind(
  get_glm_result("race", glm_nhanes_complete,
                 glm_restricted = svyglm(HI_CHOL~agecat+RIAGENDR,
                                         design=glm_nhanes_complete$survey.design,
                                         family=binomial(link="logit"))),
  get_glm_result("agecat", glm_nhanes_complete,
                 glm_restricted = svyglm(HI_CHOL~race+RIAGENDR,
                                         design=glm_nhanes_complete$survey.design,
                                         family=binomial(link="logit"))),
  get_glm_result("RIAGENDR", glm_nhanes_complete,
                 glm_restricted = svyglm(HI_CHOL~race+agecat,
                                         design=glm_nhanes_complete$survey.design,
                                         family=binomial(link="logit"))))
write_result(glm_result_nhanes_complete, 'nhanes_complete_result_r.csv')
# RIAGENDR is binary, need to change calculation to match python
glm_result_nhanes_complete <- update_binary_result(
  ewas_result = glm_result_nhanes_complete,
  new_bin_result = get_glm_result("RIAGENDR", glm_nhanes_complete, alt_name="RIAGENDR2")
  )
write_result(glm_result_nhanes_complete, 'nhanes_complete_result.csv')


# Full design: cluster, strata, weights with some categorical data missing
print("Full Design with missing")
dnhanes_complete <- svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_NAs)
glm_nhanes_complete <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=dnhanes_complete, family=binomial(link="logit"), na.action=na.omit)
glm_nhanes_complete_race <- svyglm(
    HI_CHOL~race+agecat+RIAGENDR,
    design=subset(dnhanes_complete, !is.na(dnhanes_complete$variables$race)),
    family=binomial(link="logit"),
    na.action=na.omit)
glm_result_nhanes_complete <- rbind(
  get_glm_result("race", glm_nhanes_complete_race,
                 glm_restricted = svyglm(HI_CHOL~agecat+RIAGENDR,
                                         design=glm_nhanes_complete_race$survey.design,
                                         family=binomial(link="logit"))),
  get_glm_result("agecat", glm_nhanes_complete,
                 glm_restricted = svyglm(HI_CHOL~race+RIAGENDR,
                                         design=glm_nhanes_complete$survey.design,
                                         family=binomial(link="logit"))),
  get_glm_result("RIAGENDR", glm_nhanes_complete,
                 glm_restricted = svyglm(HI_CHOL~race+agecat,
                                         design=glm_nhanes_complete$survey.design,
                                         family=binomial(link="logit"))))
write.csv(glm_result_nhanes_complete, 'nhanes_complete_withna_result_r.csv', row.names=FALSE)
# RIAGENDR is binary, need to change calculation to match python
glm_result_nhanes_complete <- update_binary_result(
  ewas_result = glm_result_nhanes_complete,
  new_bin_result = get_glm_result("RIAGENDR", glm_nhanes_complete, alt_name="RIAGENDR2")
  )
write_result(glm_result_nhanes_complete, 'nhanes_complete_withna_result.csv')

# Full design: cluster, strata, weights with subset agecat != (19,39]
print("Full Design with subset")
dnhanes_complete <- svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes)
dnhanes_subset <- subset(dnhanes_complete, agecat!="(19,39]")
glm_nhanes_complete <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=dnhanes_subset, family=binomial(link="logit"), na.action=na.omit)
glm_result_nhanes_complete <- rbind(
  get_glm_result("race", glm_nhanes_complete,
                 glm_restricted = svyglm(HI_CHOL~agecat+RIAGENDR,
                                         design=glm_nhanes_complete$survey.design,
                                         family=binomial(link="logit"))),
  get_glm_result("agecat", glm_nhanes_complete,
                 glm_restricted = svyglm(HI_CHOL~race+RIAGENDR,
                                         design=glm_nhanes_complete$survey.design,
                                         family=binomial(link="logit"))),
  get_glm_result("RIAGENDR", glm_nhanes_complete,
                 glm_restricted = svyglm(HI_CHOL~race+agecat,
                                         design=glm_nhanes_complete$survey.design,
                                         family=binomial(link="logit"))))
write.csv(glm_result_nhanes_complete, 'nhanes_complete_result_subset_r.csv', row.names=FALSE)
# RIAGENDR is binary, need to change calculation to match python
glm_result_nhanes_complete <- update_binary_result(
  ewas_result = glm_result_nhanes_complete,
  new_bin_result = get_glm_result("RIAGENDR", glm_nhanes_complete, alt_name="RIAGENDR2")
  )
write_result(glm_result_nhanes_complete, 'nhanes_complete_result_subset.csv')



# Weights Only
print("Weights Only")
dnhanes_weightsonly <- svydesign(id=~1, weights=~WTMEC2YR, data=nhanes)
glm_nhanes_weightsonly <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=dnhanes_weightsonly, family=binomial(link="logit"), na.action=na.omit)
glm_result_nhanes_weightsonly <- rbind(
  get_glm_result("race", glm_nhanes_weightsonly,
                 glm_restricted = svyglm(HI_CHOL~agecat+RIAGENDR, design=glm_nhanes_weightsonly$survey.design, family=binomial(link="logit"))),
  get_glm_result("agecat", glm_nhanes_weightsonly,
                 glm_restricted = svyglm(HI_CHOL~race+RIAGENDR, design=glm_nhanes_weightsonly$survey.design, family=binomial(link="logit"))),
  get_glm_result("RIAGENDR", glm_nhanes_weightsonly,
                 glm_restricted = svyglm(HI_CHOL~race+agecat, design=glm_nhanes_weightsonly$survey.design, family=binomial(link="logit"))))
write.csv(glm_result_nhanes_weightsonly, 'nhanes_weightsonly_result_r.csv', row.names=FALSE)
# RIAGENDR is binary, need to change calculation to match python
glm_result_nhanes_weightsonly <- update_binary_result(
  ewas_result = glm_result_nhanes_weightsonly,
  new_bin_result = get_glm_result("RIAGENDR", glm_nhanes_weightsonly, alt_name="RIAGENDR2")
  )
write_result(glm_result_nhanes_weightsonly, 'nhanes_weightsonly_result.csv')

#################
# NHANES Lonely #
#################
print("NHANES Lonely Tests")
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
  glm_nhanes_lonely <- svyglm(HI_CHOL~race+agecat+RIAGENDR,
                              design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely),
                              family=binomial(link="logit"),
                              na.action=na.omit)
  if(binary_as_continuous){
    glm_result_nhanes_lonely <- rbind(
      get_glm_result("race", glm_nhanes_lonely,
                     glm_restricted = svyglm(HI_CHOL~agecat+RIAGENDR, design=glm_nhanes_lonely$survey.design, family=binomial(link="logit"))),
      get_glm_result("agecat", glm_nhanes_lonely,
                     glm_restricted = svyglm(HI_CHOL~race+RIAGENDR, , design=glm_nhanes_lonely$survey.design, family=binomial(link="logit"))),
      get_glm_result("RIAGENDR", glm_nhanes_lonely, alt_name="RIAGENDR2"))
  } else {
    glm_result_nhanes_lonely <- rbind(
      get_glm_result("race", glm_nhanes_lonely,
                     glm_restricted = svyglm(HI_CHOL~agecat+RIAGENDR, design=glm_nhanes_lonely$survey.design, family=binomial(link="logit"))),
      get_glm_result("agecat", glm_nhanes_lonely,
                     glm_restricted = svyglm(HI_CHOL~race+RIAGENDR, design=glm_nhanes_lonely$survey.design, family=binomial(link="logit"))),
      get_glm_result("RIAGENDR", glm_nhanes_lonely,
                     glm_restricted = svyglm(HI_CHOL~race+agecat, design=glm_nhanes_lonely$survey.design, family=binomial(link="logit"))))
  }
  return(glm_result_nhanes_lonely)
}

# Certainty
glm_result_nhanes_certainty <- get_lonely_glm_results("certainty", binary_as_continuous=FALSE)
write_result(glm_result_nhanes_certainty, 'nhanes_certainty_result_r.csv')
glm_result_nhanes_certainty <- get_lonely_glm_results("certainty", binary_as_continuous=TRUE)
write_result(glm_result_nhanes_certainty, 'nhanes_certainty_result.csv')

# Adjust
glm_result_nhanes_adjust <- get_lonely_glm_results("adjust", binary_as_continuous=FALSE)
write_result(glm_result_nhanes_adjust, 'nhanes_adjust_result_r.csv')
glm_result_nhanes_adjust <- get_lonely_glm_results("adjust", binary_as_continuous=TRUE)
write_result(glm_result_nhanes_adjust, 'nhanes_adjust_result.csv')

# Average
glm_result_nhanes_adjust <- get_lonely_glm_results("average", binary_as_continuous=FALSE)
write_result(glm_result_nhanes_adjust, 'nhanes_average_result_r.csv')
glm_result_nhanes_adjust <- get_lonely_glm_results("average", binary_as_continuous=TRUE)
write_result(glm_result_nhanes_adjust, 'nhanes_average_result.csv')

#########################
# Realistic NHANES Test #
#########################
print("Realistic NHANES Test")

# Multiple weights, missing data, etc
data <- read.table("../test_data_files/nhanes_real.txt", sep="\t", header=TRUE)
data$RHQ570 <- as.factor(data$RHQ570)
data$first_degree_support <- as.factor(data$first_degree_support)
data$SDDSRVYR <- as.factor(data$SDDSRVYR)
data$female <- as.factor(data$female)
data$black <- as.factor(data$black)
data$mexican <- as.factor(data$mexican)
data$other_hispanic <- as.factor(data$other_hispanic)
data$other_eth <- as.factor(data$other_eth)
data$SES_LEVEL <- as.factor(data$SES_LEVEL)

# Missing values in weights have to be replaced with 0 to pass a check in the survey library
data[is.na(data$WTSHM4YR), "WTSHM4YR"] <- 0
data[is.na(data$WTSVOC4Y), "WTSVOC4Y"] <- 0

# RHQ570 - skip nonvarying 'female' covariate
glm_full_RHQ570 <- svyglm(as.formula(BMXBMI~SES_LEVEL+SDDSRVYR+black+mexican+other_hispanic+other_eth+RIDAGEYR+RHQ570),
                          design=subset(
                            svydesign(weights=~WTMEC4YR, ids=~SDMVPSU, strata=~SDMVSTRA, data=data, nest=TRUE),
                            !is.na(data$RHQ570)),
                          na.action=na.omit)
glm_restricted <- svyglm(as.formula(BMXBMI~SES_LEVEL+SDDSRVYR+black+mexican+other_hispanic+other_eth+RIDAGEYR),
                         design=glm_full_RHQ570$survey.design,
                         na.action=na.omit)
result_RHQ570 <- get_glm_result("RHQ570",
                                glm_full=glm_full_RHQ570,
                                glm_restricted=glm_restricted,
                                use_weights=TRUE)
# first_degree_support
glm_full_first_degree_support <- svyglm(as.formula(BMXBMI~SES_LEVEL+SDDSRVYR+female+black+mexican+other_hispanic+other_eth+RIDAGEYR+first_degree_support),
                                        design=subset(
                                          svydesign(weights=~WTMEC4YR, ids=~SDMVPSU, strata=~SDMVSTRA, data=data, nest=TRUE),
                                          !is.na(data$first_degree_support)),
                                        na.action=na.omit)
glm_restricted <- svyglm(as.formula(BMXBMI~SES_LEVEL+SDDSRVYR+female+black+mexican+other_hispanic+other_eth+RIDAGEYR),
                         design=glm_full_first_degree_support$survey.design,
                         na.action=na.omit)
result_first_degree_support <- get_glm_result("first_degree_support",
                                              glm_full_first_degree_support,
                                              glm_restricted,
                                              use_weights=TRUE)
# URXUPT
glm_full <- svyglm(as.formula(BMXBMI~SES_LEVEL+SDDSRVYR+female+black+mexican+other_hispanic+other_eth+RIDAGEYR+URXUPT),
                   design=svydesign(weights=~WTSHM4YR, ids=~SDMVPSU, strata=~SDMVSTRA, data=data, nest=TRUE),
                   na.action=na.omit)
result_URXUPT <- get_glm_result("URXUPT",
                                glm_full,
                                glm_restricted=NULL,
                                use_weights=TRUE)

# LBXV3A - skip nonvarying 'SDDSRVYR' covariate
glm_full <- svyglm(as.formula(BMXBMI~SES_LEVEL+female+black+mexican+other_hispanic+other_eth+RIDAGEYR+LBXV3A),
                   design=svydesign(weights=~WTSVOC4Y, ids=~SDMVPSU, strata=~SDMVSTRA, data=data, nest=TRUE),
                   na.action=na.omit)
result_LBXV3A <- get_glm_result("LBXV3A",
                                glm_full,
                                glm_restricted=NULL,
                                use_weights=TRUE)

# LBXBEC - skip nonvarying 'SDDSRVYR' covariate
glm_full <- svyglm(as.formula(BMXBMI~SES_LEVEL+female+black+mexican+other_hispanic+other_eth+RIDAGEYR+LBXBEC),
                   design=svydesign(weights=~WTMEC4YR, ids=~SDMVPSU, strata=~SDMVSTRA, data=data, nest=TRUE),
                   na.action=na.omit)
result_LBXBEC <- get_glm_result("LBXBEC",
                                glm_full,
                                glm_restricted=NULL,
                                use_weights=TRUE)

# Merge and save R results
result <- rbind(
    result_RHQ570,
    result_first_degree_support,
    result_URXUPT,
    result_LBXV3A,
    result_LBXBEC
)
write_result(result, 'nhanes_real_r.csv')

# Update python results (use regression directly for binary, not an LRT)
result_RHQ570_py <- update_binary_result(ewas_result = result_RHQ570,
                               new_bin_result = get_glm_result("RHQ570",
                                                               glm_full_RHQ570,
                                                               glm_restricted=NULL,
                                                               alt_name="RHQ5701"))
result_first_degree_support_py <- update_binary_result(ewas_result = result_first_degree_support,
                               new_bin_result = get_glm_result("first_degree_support",
                                                               glm_full_first_degree_support,
                                                               glm_restricted=NULL,
                                                               alt_name="first_degree_support1"))
# Save python results
result <- rbind(
    result_RHQ570_py,
    result_first_degree_support_py,
    result_URXUPT,
    result_LBXV3A,
    result_LBXBEC
)
write_result(result, 'nhanes_real_python.csv')