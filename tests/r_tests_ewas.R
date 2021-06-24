# This script loads several datasets from the survey library abnd runs GLMs to compare results in CLARITE

set.seed(1855)

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

  write.csv(data, file.path("r_test_output/analyze", filename), row.names=FALSE)
}

get_glm_result <- function(variable, glm_full, glm_restricted=NULL, use_weights=TRUE, alt_name=NULL) {
  # Gathers results from the glm for continuous variables
  # Calculates results from two glms for categorical/binary variables
  if(is.null(glm_restricted)){
    # No restricted model = continuous variable
    # If a binary variable is being tested like python, the indexed name will be different and specified by alt_name
    if(is.null(alt_name)){
      var_idx_name <- variable
      var_type <- "continuous"
    } else {
      var_idx_name <- alt_name
      var_type <- "binary"
    }
    glm_table <- summary(glm_full)$coefficients
    se <- glm_table[var_idx_name, 'Std. Error']
    if(se == Inf) {
      # No pvalue or beta will show up
      se <- NA
      beta <- NA
      pval <- 1.0
    } else {
      beta <- glm_table[var_idx_name, 'Estimate']
      # Pval is based on a t-value for gaussian and a z-value for binomial
      if ('Pr(>|t|)' %in% colnames(glm_table)){
        pval <- glm_table[var_idx_name, 'Pr(>|t|)']
      } else {
        pval <- glm_table[var_idx_name, 'Pr(>|z|)']
      }
    }
    return(data.frame("Variable"=variable,
                      "Variable_type"=var_type,
                      "N"=nobs(glm_full),
                      "Beta"=beta,
                      "SE"=se,
                      "Diff_AIC"=NA,
                      "pvalue"=pval))
  } else {
    # Restricted model exists = categorical variable (differences between stats::anova and survey::anova)
    var_type <- "categorical"
    if(use_weights){
      lrt <-  anova(glm_full, glm_restricted, method = "LRT")
      pval <- lrt$p
    } else {
      lrt <-  anova(glm_full, glm_restricted, test="LRT")
      pval <- lrt$`Pr(>Chi)`[2]
    }
    diff_aic <- glm_full$aic - glm_restricted$aic
    return(data.frame("Variable"=variable,
                      "Variable_type"=var_type,
                      "N"=nobs(glm_full),
                      "Beta"=NA,
                      "SE"=NA,
                      "Diff_AIC"=diff_aic,
                      "pvalue"=pval))
  }
}

#################
# fpc Test data #
#################
print("fpc Test data")
# One outcome (y) and one variable (x)
fpc <- read.csv(file = 'test_data_files/fpc_data.csv')

# No weights
glm_result_fpcnoweights <- rbind(get_glm_result("x", glm(y~x, data=fpc, na.action=na.omit), use_weights=FALSE))
write_result(glm_result_fpcnoweights, 'fpc_noweights_result.csv')

# Test without specifying fpc
withoutfpc <- svydesign(weights=~weight, ids=~psuid, strata=~stratid, data=fpc, nest=TRUE)
glm_result_withoutfpc <- rbind(get_glm_result("x", svyglm(y~x, design=withoutfpc, na.action=na.omit)))
write_result(glm_result_withoutfpc, 'fpc_withoutfpc_result.csv')

# Test with specifying fpc
withfpc <- svydesign(weights=~weight, ids=~psuid, strata=~stratid, fpc=~Nh, data=fpc, nest=TRUE)
glm_result_withfpc <- rbind(get_glm_result("x", svyglm(y~x, design=withfpc, na.action=na.omit)))
write_result(glm_result_withfpc, 'fpc_withfpc_result.csv')

# Test with specifying fpc and no strata
# Have to make fpc identical now that it is one strata
fpc_nostrat <- read.csv(file= "test_data_files/fpc_nostrat_data.csv")
withfpc_nostrata <- svydesign(weights=~weight, ids=~psuid, strata=NULL, fpc=~Nh, data=fpc_nostrat)
glm_result_withfpc_nostrata <- rbind(get_glm_result("x", svyglm(y~x, design=withfpc_nostrata, na.action=na.omit)))
write_result(glm_result_withfpc_nostrata, 'fpc_withfpc_nostrat_result.csv')

#################
# api Test data #
#################
print("api Test data")
# one outcome (api00) and 3 continuous variables (ell, meals, mobility)
apipop <- read.csv(file = 'test_data_files/apipop_data.csv')
apipop_withna <- read.csv(file = 'test_data_files/apipop_withna_data.csv')
apistrat <- read.csv(file = 'test_data_files/apistrat_data.csv')
apiclus1 <- read.csv(file = 'test_data_files/apiclus1_data.csv')

# Full population no weights
glm_apipop <- glm(api00~ell+meals+mobility, data=apipop, na.action=na.omit)
glm_result_apipop <- rbind(
  get_glm_result("ell", glm_apipop, use_weights = FALSE),
  get_glm_result("meals", glm_apipop, use_weights = FALSE),
  get_glm_result("mobility", glm_apipop, use_weights = FALSE)
)
write_result(glm_result_apipop, 'api_apipop_result.csv')

# Full population no weights, with NA
glm_apipop_withna <- glm(api00~ell+meals+mobility, data=apipop_withna, na.action=na.omit)
glm_result_apipop_withna <- rbind(
  get_glm_result("ell", glm_apipop_withna, use_weights = FALSE),
  get_glm_result("meals", glm_apipop_withna, use_weights = FALSE),
  get_glm_result("mobility", glm_apipop_withna, use_weights = FALSE)
)
write_result(glm_result_apipop_withna, 'api_apipop_withna_result.csv')

# stratified sample (no clusters) with fpc
dstrat <- svydesign(id=~1, strata=~stype, weights=~pw, data=apistrat, fpc=~fpc)
glm_apistrat <- svyglm(api00~ell+meals+mobility, design=dstrat, na.action=na.omit)
glm_result_apistrat <- rbind(
  get_glm_result("ell", glm_apistrat),
  get_glm_result("meals", glm_apistrat),
  get_glm_result("mobility", glm_apistrat)
)
write_result(glm_result_apistrat, 'api_apistrat_result.csv')

# one-stage cluster sample (no strata) with fpc
dclus1 <- svydesign(id=~dnum, weights=~pw, data=apiclus1, fpc=~fpc)
glm_apiclus1 <- svyglm(api00~ell+meals+mobility, design=dclus1, na.action=na.omit)
glm_result_apiclus1 <- rbind(
  get_glm_result("ell", glm_apiclus1),
  get_glm_result("meals", glm_apiclus1),
  get_glm_result("mobility", glm_apiclus1)
)
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

nhanes <- read.csv('test_data_files/nhanes_data.csv')
nhanes$HI_CHOL <- as.factor(nhanes$HI_CHOL)
nhanes$race <- as.factor(nhanes$race)
nhanes$agecat <- as.factor(nhanes$agecat)
nhanes$RIAGENDR <- as.factor(nhanes$RIAGENDR)
nhanes_subset <- read.csv('test_data_files/nhanes_data_subset.csv')
nhanes_subset$HI_CHOL <- as.factor(nhanes_subset$HI_CHOL)
nhanes_subset$race <- as.factor(nhanes_subset$race)
nhanes_subset$agecat <- as.factor(nhanes_subset$agecat)
nhanes_subset$RIAGENDR <- as.factor(nhanes_subset$RIAGENDR)

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
  get_glm_result("RIAGENDR", glm_nhanes_noweights, glm_restricted=NULL, use_weights=FALSE, alt_name="RIAGENDR2")
)
write_result(glm_result_nhanes_noweights, 'nhanes_noweights_result.csv')

# Full population no weights, with some categorical data missing
print("Full population no weights, with some categorical data missing")
# Load data
nhanes_NAs <- read.csv('test_data_files/nhanes_NAs_data.csv')
nhanes_NAs$HI_CHOL <- as.factor(nhanes_NAs$HI_CHOL)
nhanes_NAs$race <- as.factor(nhanes_NAs$race)
nhanes_NAs$agecat <- as.factor(nhanes_NAs$agecat)
nhanes_NAs$RIAGENDR <- as.factor(nhanes_NAs$RIAGENDR)

glm_nhanes_noweights <- glm(HI_CHOL~race+agecat+RIAGENDR, family=binomial(link="logit"), data=nhanes_NAs, na.action=na.omit)
glm_result_nhanes_noweights <- rbind(
  get_glm_result("race", glm_nhanes_noweights,
                 glm_restricted = glm(HI_CHOL~agecat+RIAGENDR, family=binomial(link="logit"), data=glm_nhanes_noweights$model),
                 use_weights=FALSE),
  get_glm_result("agecat", glm_nhanes_noweights,
                 glm_restricted = glm(HI_CHOL~race+RIAGENDR, family=binomial(link="logit"), data=glm_nhanes_noweights$model),
                 use_weights=FALSE),
  get_glm_result("RIAGENDR", glm_nhanes_noweights, glm_restricted=NULL, use_weights=FALSE, alt_name="RIAGENDR2")
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
  get_glm_result("RIAGENDR", glm_nhanes_complete, alt_name="RIAGENDR2")
)
write_result(glm_result_nhanes_complete, 'nhanes_complete_result.csv')

# Full design: cluster, strata, weights with some categorical data missing
print("Full Design with missing")
dnhanes_complete <- svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_NAs)
dnhanes_complete_race <- subset(dnhanes_complete, !is.na(race))
glm_nhanes_complete <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=dnhanes_complete, family=binomial(link="logit"), na.action=na.omit)
glm_nhanes_complete_race <- svyglm(
    HI_CHOL~race+agecat+RIAGENDR,
    design=dnhanes_complete_race,
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
  get_glm_result("RIAGENDR", glm_nhanes_complete, alt_name="RIAGENDR2")
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
  get_glm_result("RIAGENDR", glm_nhanes_complete, alt_name="RIAGENDR2")
)
write_result(glm_result_nhanes_complete, 'nhanes_complete_result_subset_cat.csv')

# Full design: cluster, strata, weights with about half of observations randomly selected
print("Full Design with continous subset")
dnhanes_complete <- svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_subset)
dnhanes_subset <- subset(dnhanes_complete, subset > 0)
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
  get_glm_result("RIAGENDR", glm_nhanes_complete, alt_name="RIAGENDR2")
)
write_result(glm_result_nhanes_complete, 'nhanes_complete_result_subset_cont.csv')

# Weights Only
print("Weights Only")
dnhanes_weightsonly <- svydesign(id=~1, weights=~WTMEC2YR, data=nhanes)
glm_nhanes_weightsonly <- svyglm(HI_CHOL~race+agecat+RIAGENDR, design=dnhanes_weightsonly, family=binomial(link="logit"), na.action=na.omit)
glm_result_nhanes_weightsonly <- rbind(
  get_glm_result("race", glm_nhanes_weightsonly,
                 glm_restricted = svyglm(HI_CHOL~agecat+RIAGENDR, design=glm_nhanes_weightsonly$survey.design, family=binomial(link="logit"))),
  get_glm_result("agecat", glm_nhanes_weightsonly,
                 glm_restricted = svyglm(HI_CHOL~race+RIAGENDR, design=glm_nhanes_weightsonly$survey.design, family=binomial(link="logit"))),
  get_glm_result("RIAGENDR", glm_nhanes_weightsonly, alt_name="RIAGENDR2")
)
write_result(glm_result_nhanes_weightsonly, 'nhanes_weightsonly_result.csv')

#################
# NHANES Lonely #
#################
print("NHANES Lonely Tests")
# Lonely PSU (only one PSU in a stratum)
# Make Lonely PSUs by dropping some rows

nhanes_lonely <- read.csv('test_data_files/nhanes_lonely_data.csv')
nhanes_lonely$HI_CHOL <- as.factor(nhanes_lonely$HI_CHOL)
nhanes_lonely$race <- as.factor(nhanes_lonely$race)
nhanes_lonely$agecat <- as.factor(nhanes_lonely$agecat)
nhanes_lonely$RIAGENDR <- as.factor(nhanes_lonely$RIAGENDR)


get_lonely_glm_results <- function(setting){
  options(survey.lonely.psu=setting)
  glm_nhanes_lonely <- svyglm(HI_CHOL~race+agecat+RIAGENDR,
                              design=svydesign(id=~SDMVPSU, strata=~SDMVSTRA, weights=~WTMEC2YR, nest=TRUE, data=nhanes_lonely),
                              family=binomial(link="logit"),
                              na.action=na.omit)
  glm_result_nhanes_lonely <- rbind(
      get_glm_result("race", glm_nhanes_lonely,
                     glm_restricted = svyglm(HI_CHOL~agecat+RIAGENDR, design=glm_nhanes_lonely$survey.design, family=binomial(link="logit"))),
      get_glm_result("agecat", glm_nhanes_lonely,
                     glm_restricted = svyglm(HI_CHOL~race+RIAGENDR, design=glm_nhanes_lonely$survey.design, family=binomial(link="logit"))),
      get_glm_result("RIAGENDR", glm_nhanes_lonely, alt_name="RIAGENDR2")
  )
  return(glm_result_nhanes_lonely)
}

# Certainty
glm_result_nhanes_certainty <- get_lonely_glm_results("certainty")
write_result(glm_result_nhanes_certainty, 'nhanes_certainty_result.csv')

# Adjust
glm_result_nhanes_adjust <- get_lonely_glm_results("adjust")
write_result(glm_result_nhanes_adjust, 'nhanes_adjust_result.csv')

# Average
glm_result_nhanes_adjust <- get_lonely_glm_results("average")
write_result(glm_result_nhanes_adjust, 'nhanes_average_result.csv')

#########################
# Realistic NHANES Test #
#########################
print("Realistic NHANES Test")

# Multiple weights, missing data, etc
data <- read.table("test_data_files/nhanes_real.txt", sep="\t", header=TRUE)
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
                          design=svydesign(weights=~WTMEC4YR, ids=~SDMVPSU, strata=~SDMVSTRA, data=data, nest=TRUE),
                          na.action=na.omit)
result_RHQ570 <- get_glm_result("RHQ570", glm_full_RHQ570, glm_restricted=NULL, alt_name="RHQ5701")

# first_degree_support
glm_full_first_degree_support <- svyglm(as.formula(BMXBMI~SES_LEVEL+SDDSRVYR+female+black+mexican+other_hispanic+other_eth+RIDAGEYR+first_degree_support),
                                        design=svydesign(weights=~WTMEC4YR, ids=~SDMVPSU, strata=~SDMVSTRA, data=data, nest=TRUE),
                                        na.action=na.omit)
result_first_degree_support <- get_glm_result("first_degree_support",
                                              glm_full_first_degree_support,
                                              glm_restricted=NULL,
                                              alt_name="first_degree_support1")
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
write_result(result, 'nhanes_real_result.csv')

#######################
# NHANES Subset Test #
######################
print("NHANES Subset Test")

# Load Data
data <- read.table("test_data_files/nhanes_subset/data.txt", sep="\t", header=TRUE)
survey_data <- read.table("test_data_files/nhanes_subset/design_data.txt", sep="\t", header=TRUE)
data$LBXHBC <- as.factor(data$LBXHBC)
data$black <- as.factor(data$black)
data$female <- as.factor(data$female)
data$SES_LEVEL <- as.factor(data$SES_LEVEL)
data$SDDSRVYR <- as.factor(data$SDDSRVYR)
data <- merge(data, survey_data)

# Make and subset design
design <- svydesign(weights=~WTMEC4YR, ids=~SDMVPSU, strata=~SDMVSTRA, data=data, nest=TRUE)
design <- subset(design, data$black==1)

# LBXHBC
glm_full_LBXHBC <- svyglm(as.formula(LBXLYPCT~LBXHBC+female+SES_LEVEL+RIDAGEYR+SDDSRVYR+BMXBMI),
                          design=design,
                          na.action=na.omit)
result_LBXHBC <- get_glm_result("LBXHBC", glm_full_LBXHBC, glm_restricted=NULL, alt_name="LBXHBC1")

# LBXVCF
glm_full_LBXVCF <- svyglm(as.formula(LBXLYPCT~LBXVCF+female+SES_LEVEL+RIDAGEYR+SDDSRVYR+BMXBMI),
                          design=design,
                          na.action=na.omit)
result_LBXVCF <- get_glm_result("LBXVCF",
                                glm_full=glm_full_LBXVCF,
                                glm_restricted=NULL,
                                use_weights=TRUE)
# SMD160
glm_full_SMD160 <- svyglm(as.formula(LBXLYPCT~SMD160+female+SES_LEVEL+RIDAGEYR+SDDSRVYR+BMXBMI),
                          design=design,
                          na.action=na.omit)
result_SMD160 <- get_glm_result("SMD160",
                                glm_full=glm_full_SMD160,
                                glm_restricted=NULL,
                                use_weights=TRUE)
# LBDEONO
glm_full_LBDEONO <- svyglm(as.formula(LBXLYPCT~LBDEONO+female+SES_LEVEL+RIDAGEYR+SDDSRVYR+BMXBMI),
                          design=design,
                          na.action=na.omit)
result_LBDEONO <- get_glm_result("LBDEONO",
                                glm_full=glm_full_LBDEONO,
                                glm_restricted=NULL,
                                use_weights=TRUE)
# Merge and save R results
result <- rbind(
    result_LBXHBC,
    result_LBXVCF,
    result_SMD160,
    result_LBDEONO
)
write_result(result, 'nhanes_subset_result.csv')
