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
  if(is.numeric(data$Diff_AIC)){data$Beta_pvalue <- formatC(as.numeric(data$Beta_pvalue), format = "e", digits = 6)}
  if(is.numeric(data$pvalue)){data$LRT_pvalue  <- formatC(as.numeric(data$LRT_pvalue), format = "e", digits = 6)}

  # Fix where "NA" was modified by the above
  data <- replace(data, data=="     NA", "NA")

  write.csv(data, file.path("r_test_output/interactions", filename), row.names=TRUE)
}

get_interaction_result <- function(test_num, test_name, data, family, formula_full, formula_restricted, report_betas) {
  glm_full <- glm(formula_full, data=data, family=family, na.action=na.omit)
  glm_restricted <- glm(formula_restricted, data=data, family=family, na.action=na.omit)
  if (!(glm_full$converged) | !(glm_restricted$converged)){
    return(data.frame("Test_Number"=test_num,
                      "Converged"=FALSE,
                      "N"=nobs(glm_full),
                      "Beta"=NaN,
                      "SE"=NaN,
                      "Beta_pvalue"=NaN,
                      "LRT_pvalue"=NaN))
  } else {

  }
  if(report_betas){
    lrt <-  anova(glm_full, glm_restricted, test="LRT")
    pval <- lrt$`Pr(>Chi)`[2]
    # Get beta test results
    glm_table <- summary(glm_full)$coefficients
    idx <- grepl(':', rownames(glm_table))
    beta <- glm_table[idx, 'Estimate']
    se <- glm_table[idx, 'Std. Error']
    beta_pval <- glm_table[idx, 'Pr(>|z|)']
    return(data.frame("Test_Number"=rep(test_num, times=sum(idx)),
                      "Converged"=rep(TRUE, times=sum(idx)),
                      "N"=rep(nobs(glm_full), times=sum(idx)),
                      "Beta"=beta,
                      "SE"=se,
                      "Beta_pvalue"=beta_pval,
                      "LRT_pvalue"=rep(pval, times=sum(idx))))
  } else {
    lrt <-  anova(glm_full, glm_restricted, test="LRT")
    pval <- lrt$`Pr(>Chi)`[2]
    result<- data.frame("Test_Number"=test_num,
                      "Converged"=TRUE,
                      "N"=nobs(glm_full),
                      "Beta"=NaN,
                      "SE"=NaN,
                      "Beta_pvalue"=NaN,
                      "LRT_pvalue"=pval)
    rownames(result) <- test_name
    return(result)
  }
}


##########
# NHANES #
##########
nhanes <- read.csv('test_data_files/nhanes_data.csv')
nhanes$HI_CHOL <- as.factor(nhanes$HI_CHOL)
nhanes$race <- as.factor(nhanes$race)
nhanes$agecat <- as.factor(nhanes$agecat)
nhanes$RIAGENDR <- as.factor(nhanes$RIAGENDR)

#test_interactions_nhanes_ageXgender
data <- nhanes[!is.na(nhanes$HI_CHOL), ]
result_nobeta <- get_interaction_result(1,
                                        "agecat:RIAGENDR",
                                        data,
                                        family=binomial(link="logit"),
                                        HI_CHOL~race+agecat:RIAGENDR,
                                        HI_CHOL~race,
                                        FALSE)
write_result(result_nobeta, "nhanes_ageXgender_nobetas.csv")
result <- get_interaction_result(1,
                                 "agecat:RIAGENDR",
                                 data,
                                 family=binomial(link="logit"),
                                 HI_CHOL~race+agecat:RIAGENDR,
                                 HI_CHOL~race,
                                 TRUE)
write_result(result, "nhanes_ageXgender.csv")


#test_interactions_nhanes_pairwise
data <- nhanes[!is.na(nhanes$HI_CHOL), ]
result_nobeta <- rbind(
  get_interaction_result(1,
                         "agecat:RIAGENDR",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~agecat:RIAGENDR,
                         HI_CHOL~1,
                         FALSE),
  get_interaction_result(2,
                         "race:agecat",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~race:agecat,
                         HI_CHOL~1,
                         FALSE),
  get_interaction_result(3,
                         "RIAGENDR:race",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~RIAGENDR:race,
                         HI_CHOL~1,
                         FALSE)
)
write_result(result_nobeta, "nhanes_pairwise_nobetas.csv")
result <- rbind(
  get_interaction_result(1,
                         "agecat:RIAGENDR",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~agecat:RIAGENDR,
                         HI_CHOL~1,
                         TRUE),
  get_interaction_result(2,
                         "race:agecat",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~race:agecat,
                         HI_CHOL~1,
                         TRUE),
  get_interaction_result(3,
                         "RIAGENDR:race",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~RIAGENDR:race,
                         HI_CHOL~1,
                         TRUE)
)
write_result(result, "nhanes_pairwise.csv")