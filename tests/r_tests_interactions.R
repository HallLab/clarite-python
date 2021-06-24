# This script loads several datasets from the survey library abnd runs GLMs to compare results in CLARITE

set.seed(1855)

library(devtools)
if (!require('survey')) install.packages('survey', repos = "http://cran.us.r-project.org"); library('survey')

####################
# Useful Functions #
####################

write_result <- function(data, filename, outcome="HI_CHOL") {
  # Round numeric columns to limit precision
  if(is.numeric(data$Beta)){data$Beta <- formatC(data$Beta, format = "e", digits = 6)}
  if(is.numeric(data$SE)){data$SE <- formatC(as.numeric(data$SE), format = "e", digits = 6)}
  if(is.numeric(data$Diff_AIC)){data$Beta_pvalue <- formatC(as.numeric(data$Beta_pvalue), format = "e", digits = 6)}
  if(is.numeric(data$pvalue)){data$LRT_pvalue  <- formatC(as.numeric(data$LRT_pvalue), format = "e", digits = 6)}

  # Fix where "NA" was modified by the above
  data <- replace(data, data=="     NA", "NA")

  # Add outcome
  data$Outcome <- outcome
  
  # Save
  write.csv(data, file.path("r_test_output/interactions", filename), row.names=FALSE)
}

get_interaction_result <- function(term1, term2, data, family, formula_full, formula_restricted, report_betas=FALSE) {
  glm_full <- glm(formula_full, data=data, family=family, na.action=na.omit)
  glm_restricted <- glm(formula_restricted, data=data, family=family, na.action=na.omit)
  if (!(glm_full$converged) | !(glm_restricted$converged)){
    return(data.frame("Term1"=term1,
                      "Term2"=term2,
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
    names_full <- rownames(glm_table)
    names_restricted <- rownames(summary(glm_restricted)$coefficients)
    idx <- setdiff(names_full, names_restricted)
    beta <- glm_table[idx, 'Estimate']
    se <- glm_table[idx, 'Std. Error']
    beta_pval <- glm_table[idx, 'Pr(>|z|)']
    return(data.frame("Term1"=rep(term1, times=length(idx)),
                      "Term2"=rep(term2, times=length(idx)),
                      "Parameter"=idx,
                      "Converged"=rep(TRUE, times=length(idx)),
                      "N"=rep(nobs(glm_full), times=length(idx)),
                      "Beta"=beta,
                      "SE"=se,
                      "Beta_pvalue"=beta_pval,
                      "LRT_pvalue"=rep(pval, times=length(idx))))
  } else {
    lrt <-  anova(glm_full, glm_restricted, test="LRT")
    pval <- lrt$`Pr(>Chi)`[2]
    result<- data.frame("Term1"=term1,
                        "Term2"=term2,
                        "Converged"=TRUE,
                        "N"=nobs(glm_full),
                        "Beta"=NaN,
                        "SE"=NaN,
                        "Beta_pvalue"=NaN,
                        "LRT_pvalue"=pval)
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

# Use HI_CHOL as output, dropping NA values
data <- nhanes[!is.na(nhanes$HI_CHOL), ]

# test_interaction_agecatXRIAGENDR
result_nobeta <- get_interaction_result("agecat",
                                        "RIAGENDR",
                                        data,
                                        family=binomial(link="logit"),
                                        HI_CHOL~1+race+agecat+RIAGENDR+agecat:RIAGENDR,
                                        HI_CHOL~1+race+agecat+RIAGENDR)
write_result(result_nobeta, "nhanes_ageXgender.csv")

result <- get_interaction_result("agecat",
                                 "RIAGENDR",
                                 data,
                                 family=binomial(link="logit"),
                                 HI_CHOL~race+agecat+RIAGENDR+agecat:RIAGENDR,
                                 HI_CHOL~race+agecat+RIAGENDR,
                                 report_betas = TRUE)
write_result(result, "nhanes_ageXgender_withbetas.csv")

# test_interaction_weightXrace
result_nobeta <- get_interaction_result("WTMEC2YR",
                                        "race",
                                        data,
                                        family=binomial(link="logit"),
                                        HI_CHOL~agecat+RIAGENDR+WTMEC2YR+race+WTMEC2YR:race,
                                        HI_CHOL~agecat+RIAGENDR+WTMEC2YR+race)
write_result(result_nobeta, "nhanes_weightXrace.csv")

result <- get_interaction_result("WTMEC2YR",
                                 "race",
                                 data,
                                 family=binomial(link="logit"),
                                 HI_CHOL~agecat+RIAGENDR+WTMEC2YR+race+WTMEC2YR:race,
                                 HI_CHOL~agecat+RIAGENDR+WTMEC2YR+race,
                                 report_betas = TRUE)
write_result(result, "nhanes_weightXrace_withbetas.csv")


#test_interactions_nhanes_pairwise

result_nobeta <- rbind(
  get_interaction_result("RIAGENDR",
                         "agecat",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~RIAGENDR+agecat+RIAGENDR:agecat,
                         HI_CHOL~RIAGENDR+agecat),
  get_interaction_result("race",
                         "agecat",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~race+agecat+race:agecat,
                         HI_CHOL~race+agecat),
  get_interaction_result("RIAGENDR",
                         "race",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~RIAGENDR+race+RIAGENDR:race,
                         HI_CHOL~RIAGENDR+race)
)
write_result(result_nobeta, "nhanes_pairwise.csv")

result <- rbind(
  get_interaction_result("RIAGENDR",
                         "agecat",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~RIAGENDR+agecat+RIAGENDR:agecat,
                         HI_CHOL~RIAGENDR+agecat,
                         report_betas = TRUE),
  get_interaction_result("race",
                         "agecat",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~race+agecat+race:agecat,
                         HI_CHOL~race+agecat,
                         report_betas = TRUE),
  get_interaction_result("RIAGENDR",
                         "race",
                         data,
                         family=binomial(link="logit"),
                         HI_CHOL~RIAGENDR+race+RIAGENDR:race,
                         HI_CHOL~RIAGENDR+race,
                         report_betas = TRUE)
)
write_result(result, "nhanes_pairwise_withbetas.csv")