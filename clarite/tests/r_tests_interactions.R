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

  write.csv(data, file.path("r_test_output/interactions", filename), row.names=FALSE)
}

get_interaction_result <- function(data, i1, i2, covariates, report_betas) {
    return(data.frame("Test_Number"=1,
                      "Converged"=TRUE,
                      "N"=nobs(glm_full),
                      "Beta"=beta,
                      "SE"=se,
                      "Beta_pvalue"=NA,
                      "LRT_pvalue"=pval))
}