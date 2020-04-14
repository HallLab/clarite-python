library(survey)

# Catch errors from glm and similar, warning instead
warn_on_e <- function(var_name, e){
  warning(paste0("NULL result for ", var_name, " due to: ", e), call=FALSE)
  return(NULL)
}

# Get required data for regressing a specific variable
get_varying_covariates <- function(df, covariates, phenotype, variable, allowed_nonvarying){
  # Get number of unique values in covariates among observations where the variable is not NA
  cov_counts <- sapply(covariates, function(c) {length(unique(df[!is.na(df[c]) & !is.na(df[variable]), c]))})
  varying_covariates <- covariates[cov_counts >= 2]
  nonvarying_covariates <- covariates[cov_counts <2]
  # Compare to the covariates that are allowed to vary
  not_allowed_nonvarying <- setdiff(nonvarying_covariates, allowed_nonvarying)
  if(length(not_allowed_nonvarying) > 0){
    # Null Result
    print(paste0("    NULL result: Some covariates don't vary when '", variable, "' is not NA and aren't specified as allowed: ",
                 paste(not_allowed_nonvarying, collapse = ", ")))
     return(NULL)
  } else if(length(nonvarying_covariates) > 0){
    # Ignore those
    print(paste0("    Some covariates don't vary when '", variable, "' is not NA but are allowed to do so: ",
                 paste(nonvarying_covariates, collapse = ", ")))
  }
  # Return the list of covariates that are kept
  return(varying_covariates)
}

###Continuous###
regress_cont <- function(d, covariates, phenotype, var_name, regression_family, allowed_nonvarying){

  # Determine if a weighted ewas is being run
  if(class(d)[1] == "data.frame"){
    use_survey <- FALSE
  } else if(class(d)[2] == "survey.design") {
    use_survey <- TRUE
  }

  # Check Covariates and subset the data to use only observations where the variable is not NA
  if (use_survey){
    varying_covariates <- get_varying_covariates(d$variables, covariates, phenotype, var_name, allowed_nonvarying)
    #subset_data <- subset(d, !is.na(d$variables[var_name]))  # Use the survey subset function
  } else {
    varying_covariates <- get_varying_covariates(d, covariates, phenotype, var_name, allowed_nonvarying)
    #subset_data <- d[!is.na(d[var_name]),]  # use a subset of the data directly
  }

  # Return null if 'get_varying_covarites' returned NULL (b/c it found a nonvarying covariate the wasn't allowed)
  if (is.null(varying_covariates)){
    return()
  }

  # Create a regression formula
  if(length(varying_covariates)>0){
    fmla <- paste(phenotype, "~", var_name, "+", paste(varying_covariates, collapse="+"), sep="")
  } else {
    fmla <- paste(phenotype, "~", var_name, sep="")
  }

  # Run GLM
  if(use_survey){
    # Update scope of the regression_family and subset_data variables (surveyglm doesn't handle this well)
    regression_family <<- regression_family
    d <<- d
    # Use survey::svyglm
    var_result <- tryCatch(survey::svyglm(stats::as.formula(fmla), family=regression_family, design=d, na.action=na.omit),
                           error=function(e) warn_on_e(var_name, e))
  } else {
    # Use stats::glm
    var_result <- tryCatch(glm(stats::as.formula(fmla), family=regression_family, data=d, na.action=na.omit),
                           error=function(e) warn_on_e(var_name, e))
  }
  # Collect Results
  if (!is.null(var_result)){
    var_summary <- summary(var_result)
    # Update with processed summary results
    # Assume non-convergence if no p values are generated
    num_coeff_cols <- length(var_summary$coefficients)/nrow(var_summary$coefficients)
    if (num_coeff_cols < 4){
      return(data.frame(
        N = length(var_result$residuals),
        Converged = FALSE))
    } else {
      return(data.frame(
        N = length(var_result$residuals),
        Converged = TRUE,
        Beta = var_summary$coefficients[2,1],
        SE = var_summary$coefficients[2,2],
        Variable_pvalue = var_summary$coefficients[2,4],
        pval = var_summary$coefficients[2,4]
      ))
    }
  } else{
    return()
  }
}

###Categorical###
# Note categorical is trickier since the difference between survey and data.frame is more extensive than using a different function
regress_cat <- function(d, covariates, phenotype, var_name, regression_family, allowed_nonvarying){

  # Determine if a weighted ewas is being run
  if(class(d)[1] == "data.frame"){
    use_survey <- FALSE
  } else if(class(d)[2] == "survey.design") {
    use_survey <- TRUE
  }

  # Check Covariates and subset the data to use only observations where the variable is not NA
  if (use_survey){
    # The GLM function in R can handle this
    varying_covariates <- get_varying_covariates(d$variables, covariates, phenotype, var_name, allowed_nonvarying)
    #subset_data <- subset(d, !is.na(d$variables[var_name]))  # Use the survey subset function
  } else {
    varying_covariates <- get_varying_covariates(d, covariates, phenotype, var_name, allowed_nonvarying)
    #subset_data <- d[!is.na(d[var_name]),]  # use a subset of the data directly
  }

  # Return null if 'get_varying_covarites' returned NULL (b/c it found a nonvarying covariate the wasn't allowed)
  if (is.null(varying_covariates)){
    return()
  }

  # Create a regression formula and a restricted regression formula
  if(length(varying_covariates)>0){
    fmla <- paste(phenotype, "~", var_name, "+", paste(varying_covariates, collapse="+"), sep="")
    fmla_restricted <- paste(phenotype, "~", paste(varying_covariates, collapse="+"), sep="")
  } else {
    fmla <- paste(phenotype, "~", var_name, sep="")
    fmla_restricted <- paste(phenotype, "~1", sep="")
  }
  # Run GLM Functions
  if(use_survey){
    # Update scope of the family and subset_data variables (surveyglm doesn't handle this well)
    regression_family <<- regression_family
    d <<- d
    # Results using surveyglm
    print(survey::svyglm(stats::as.formula(fmla), family=regression_family, design=d, na.action=na.omit))
    var_result <- tryCatch(survey::svyglm(stats::as.formula(fmla), family=regression_family, design=d, na.action=na.omit),
                           error=function(e) warn_on_e(var_name, e))
    restricted_result <- tryCatch(survey::svyglm(stats::as.formula(fmla_restricted), family=regression_family, design=d, na.action=na.omit),
                                  error=function(e) warn_on_e(var_name, e))
    print(summary(var_result))
    if(!is.null(var_result) & !is.null(restricted_result)){
      # Get the LRT using anova
      lrt <- anova(var_result, restricted_result, method = "LRT")
      return(data.frame(
        N = length(var_result$residuals),
        Converged = var_result$converged,
        LRT_pvalue = lrt$p,
        Diff_AIC = var_result$aic - restricted_result$aic,
        pval = lrt$p
      ))
    }
  } else {
    # Results using data.frame with stats::anova
    var_result <- tryCatch(glm(stats::as.formula(fmla), family=regression_family, data=d, na.action=na.omit),
                           error=function(e) warn_on_e(var_name, e))
    restricted_result <- tryCatch(glm(stats::as.formula(fmla_restricted), family=regression_family, data=d, na.action=na.omit),
                                  error=function(e) warn_on_e(var_name, e))
    if(!is.null(var_result) & !is.null(restricted_result)){
      # Get the LRT using anova
      lrt <- anova(var_result, restricted_result, test = "LRT")
      return(data.frame(
        N = length(var_result$residuals),
        Converged = var_result$converged,
        LRT_pvalue = lrt$`Pr(>Chi)`[2],
        Diff_AIC = var_result$aic - restricted_result$aic,
        pval = lrt$`Pr(>Chi)`[2]
      ))
    }
  }
  return()
}

#' ewas
#'
#' Run environment-wide association study using svydesign from the survey package
#' @param d data.frame containing all of the data
#' @param cat_vars List of variables to regress that are categorical or binary
#' @param cont_vars  List of variables to regress that are continuous
#' @param y name(s) of response variable(s)
#' @param cat_covars List of covariates that are categorical or binary
#' @param cont_covars List of covariates that are continuous
#' @param regression_family family for the regression model as specified in glm ('gaussian' by default)
#' @param allowed_nonvarying list of covariates that are excluded from the regression when they do not vary instead of returning a NULL result.
#' @param weights NULL by default (for unweighted).  May be set to a string name of a single weight to use for every variable, or a named list that maps variable names to the weights that should be used for that variable's regression
#' @param ... other arguments passed to svydesign (like "id" or "strat") which are ignored if 'weights' is NULL
#' @return data frame containing following fields Variable, Sample Size, Converged, SE, Beta, Variable p-value, LRT, AIC, pval, phenotype, weight
#' @export
#' @family analysis functions
#' @examples
#' \dontrun{
#' ewas(d, cat_vars, cont_vars, y, cat_covars, cont_covars, regression_family)
#' }
ewas <- function(d, cat_vars=NULL, cont_vars=NULL, y, cat_covars=NULL, cont_covars=NULL,
                 regression_family="gaussian", allowed_nonvarying=NULL, weights=NULL, ...){
  # Record start time
  t1 <- Sys.time()

  # Validate inputs
  #################
  if(missing(y)){
    stop("Please specify either 'continuous' or 'categorical' type for predictor variables")
  }
  if(missing(regression_family)){
    stop("Please specify family type for glm()")
  }
  if(is.null(cat_vars)){
    cat_vars <- list()
  }
  if(is.null(cont_vars)){
    cont_vars <- list()
  }
  if(is.null(cat_covars)){
    cat_covars <- list()
  }
  if(is.null(cont_covars)){
    cont_covars <- list()
  }
  if(is.null(allowed_nonvarying)){
    allowed_nonvarying <- list()
  }

  # Ignore the covariates, phenotype, and ID if they were included in the variable lists
  remove <- c(y, cat_covars, cont_covars, "ID")
  cat_vars <- setdiff(cat_vars, remove)
  cont_vars <- setdiff(cont_vars, remove)
  # Ignore the phenotype, and ID if they were included in the covariates lists
  remove <- c(y, "ID")
  cat_covars <- setdiff(cat_covars, remove)
  cont_covars <- setdiff(cont_covars, remove)

  # Ensure variables/covariates aren't listed as multiple different types
  both <- intersect(cat_covars, cont_covars)
  if (length(both) > 0){stop("Some covariates are listed as both categorical and continuous: ", paste(both, collapse=", "))}
  both <- intersect(cat_vars, cont_vars)
  if (length(both) > 0){stop("Some variables are listed as both categorical and continuous: ", paste(both, collapse=", "))}

  # Check data
  if(class(d)[1] != "data.frame"){
    stop("Data must be a data.frame object")
  }

  # Check weights
  if(is.null(weights)){
    print("Running EWAS without a survey design adjustment")
  } else if(class(weights) == "character"){
    single_weight <- TRUE
    if(!(weights %in% names(d))){
      stop(paste(weights, "was specified as the weight, but was not found in the dataframe", sep=" "))
    }
    print("Running EWAS with a single weight used for all variables")
  } else if(class(weights) == "list"){
    single_weight <- FALSE
    print("Running EWAS with specific weights assigned for each variable")
  } else {
    stop("weights must be a string or a list")
  }

  #Correct the types and check for IDs
  #####################################
  # ID
  if(is.element('ID', names(d))==FALSE){stop("Please add ID to the data as column 1")}
  d$ID <- factor(d$ID)
  # Categorical
  if(length(cat_vars) > 0){d[cat_vars] <- lapply(d[cat_vars], factor)}
  if(length(cat_covars) > 0){d[cat_covars] <- lapply(d[cat_covars], factor)}
  # Continuous
  if(length(cont_vars) > 0){
    if(sum(sapply(d[cont_vars], is.numeric))!=length(cont_vars)){
      # TODO: This isn't right
      non_numeric_cont_vars <- setdiff(cont_vars, names(d[sapply(d, is.numeric)]))
      stop("Some continuous variables are not numeric: ", paste(non_numeric_cont_vars, collapse=", "))
    }
  }
  if (length(cont_covars) > 0){
    if(sum(sapply(d[cont_covars], is.numeric))!=length(cont_covars)){
      non_numeric_cont_covars <- setdiff(cont_covars, names(d[sapply(d, is.numeric)]))
      stop("Some continuous covariates are not numeric: ", paste(non_numeric_cont_covars, collapse=", "))
    }
  }

  # Get a combined list of covariates
  covariates <- c(cat_covars, cont_covars)

  # Run Regressions
  #################

  # Create a placeholder dataframe for results, anything not updated will be NA
  n <- length(cat_vars) + length(cont_vars)
  ewas_result_df <- data.frame(Variable = character(n),
                              N = numeric(n),
                              Converged = logical(n),
                              Beta = numeric(n),
                              SE = numeric(n),
                              Variable_pvalue = numeric(n),
                              LRT_pvalue = numeric(n),
                              Diff_AIC = numeric(n),
                              pval = numeric(n),
                              phenotype = character(n),
                              weight = character(n),
                              stringsAsFactors = FALSE
  )
  ewas_result_df[] <- NA  # Fill df with NA values
  i = 0 # Increment before processing each variable
  
  # Process categorical variables, if any
  print(paste("Processing ", length(cat_vars), " categorical variables", sep=""))
  for(var_name in cat_vars){
    # Get new row in the results
    i <- i + 1
    # Update var name and phenotype
    ewas_result_df$Variable[i] <- var_name
    ewas_result_df$phenotype <- y
    
    # Run Regression for the single variable
    if(is.null(weights)){
      result <- regress_cat(d, covariates, phenotype=y, var_name, regression_family, allowed_nonvarying)
    } else {
      # Get weight
      if(single_weight){
        weight <- weights
      } else {
        weight = weights[[var_name]]
      }
      # Record weight, moving on if no weight was provided for the variable
      if(is.null(weight)){
        next
      } else if(!(weight %in% names(d))){
        next
      }
      ewas_result_df$weight[i] <- weight
      # Create survey design object
      sd <- survey::svydesign(weights = d[weight],
                              data = d,
                              ...)
      # Regress, updating the dataframe if results were returned
      result <- regress_cat(sd, covariates, phenotype=y, var_name, regression_family, allowed_nonvarying)
    }
    
    # Save results
    if(!is.null(result)){
        ewas_result_df[i, c("N", "Converged", "LRT_pvalue", "Diff_AIC", "pval")] <- result
    }
  }

  # Process continuous variables, if any
  print(paste("Processing ", length(cont_vars), " continuous variables", sep=""))
  for(var_name in cont_vars){
    # Get new row in the results
    i <- i + 1
    # Update var name and phenotype
    ewas_result_df$Variable[i] <- var_name
    ewas_result_df$phenotype <- y
    
    # Run regression for the single variable
    if(is.null(weights)){
      result <- regress_cont(d, covariates, phenotype=y, var_name, regression_family, allowed_nonvarying)
    } else {
      # Get weight
      if(single_weight){
        weight <- weights
      } else {
        weight = weights[[var_name]]
      }
      # Record weight, moving on if no weight was provided for the variable
      if(is.null(weight)){
        next
      } else if(!(weight %in% names(d))){
        next
      }
      ewas_result_df$weight[i] <- weight
      # Create survey design object
      sd <- survey::svydesign(weights = d[weight],
                              data = d,
                              ...)
      # Regress, updating the dataframe if results were returned
      result <- regress_cont(sd, covariates, phenotype=y, var_name, regression_family, allowed_nonvarying)
    }
    if(!is.null(result)){
      ewas_result_df[i, c("N", "Converged", "Beta", "SE", "Variable_pvalue", "pval")] <- result
    }
  }

  t2 <- Sys.time()
  print(paste("Finished in", round(as.numeric(difftime(t2,t1, units="secs")), 6), "secs", sep=" "))
  n_null_results <- sum(is.null(ewas_result_df$pval))
  if (n_null_results > 0){
    warning(paste(n_null_results, "of", nrow(ewas_result_df), "variables had a NULL result due to an error (see earlier warnings for details)"))
  }

  # Sort by pval
  ewas_result_df <- ewas_result_df[order(ewas_result_df$pval),]

  # Replace NA with 'None' for correct conversion back to Pandas format
  ewas_result_df[is.na(ewas_result_df)]='None'

  return(ewas_result_df)
}