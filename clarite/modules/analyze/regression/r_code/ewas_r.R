library(survey)

# Catch errors from glm and similar, warning instead
warn_on_e <- function(var_name, e){
  warning(paste0("NULL result for ", var_name, " due to: ", e), call=FALSE)
  return(NULL)
}

# Quote variable names with backticks to account for special characters
quote_name <- function(s){paste0("`", s, "`")}

# Get required data for regressing a specific variable
get_varying_covariates <- function(df, covariates, variable, allowed_nonvarying){
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
regress_cont <- function(data, varying_covariates, outcome, var_name, regression_family){

  # Create a regression formula
  if(length(varying_covariates)>0){
    fmla <- paste(quote_name(outcome), "~", quote_name(var_name), "+", paste(lapply(varying_covariates, quote_name), collapse="+"), sep="")
  } else {
    fmla <- paste(quote_name(outcome), "~", quote_name(var_name), sep="")
  }

  var_result <- tryCatch(glm(stats::as.formula(fmla),
                             family=regression_family,
                             data=data,
                             na.action=na.omit),
                         error=function(e) warn_on_e(var_name, e))
  # Collect Results
  if (!is.null(var_result)){
    var_summary <- summary(var_result)
    # Update with processed summary results
    # Assume non-convergence if no p values are generated
    num_coeff_cols <- length(var_summary$coefficients)/nrow(var_summary$coefficients)
    if (num_coeff_cols < 4){
      return(NULL)
    } else {
      return(data.frame(
        Converged = TRUE,
        Beta = var_summary$coefficients[2,1],
        SE = var_summary$coefficients[2,2],
        Beta_pvalue = var_summary$coefficients[2,4],
        pval = var_summary$coefficients[2,4]
      ))
    }
  } else{
    return(NULL)
  }
}

regress_cont_survey <- function(data, varying_covariates, outcome, var_name, regression_family,
                                weight_values, strata_values, fpc_values, id_values, subset_array, ...){

  # Create survey design
  if(is.null(id_values)){
    survey_design <- survey::svydesign(ids = ~1,
                                       weights = weight_values,
                                       data = data,
                                       strata = strata_values,
                                       fpc = fpc_values, ...)
  } else{
    survey_design <- survey::svydesign(ids = id_values,
                                       weights = weight_values,
                                       data = data,
                                       strata = strata_values,
                                       fpc = fpc_values, ...)
  }

  # Update subset array to drop NA values of the outcome variable and subset the survey design
  survey_design <- subset(survey_design, subset_array)

  # Create a regression formula
  if(length(varying_covariates)>0){
    fmla <- paste(quote_name(outcome), "~", quote_name(var_name), "+", paste(lapply(varying_covariates, quote_name), collapse="+"), sep="")
  } else {
    fmla <- paste(quote_name(outcome), "~", quote_name(var_name), sep="")
  }

  var_result <- tryCatch(survey::svyglm(stats::as.formula(fmla), survey_design, family=regression_family, na.action=na.omit),
                         error=function(e) warn_on_e(var_name, e))
  # Collect Results
  if (!is.null(var_result)){
    var_summary <- summary(var_result)
    # Update with processed summary results
    num_coeff_cols <- length(var_summary$coefficients)/nrow(var_summary$coefficients)
    if (num_coeff_cols < 2){
      # Assume non-convergence if no p values are generated
      return(NULL)
    } else if (num_coeff_cols == 2) {
      return(data.frame(
        Converged = TRUE,
        Beta = var_summary$coefficients[2,1],
        SE = var_summary$coefficients[2,2],
        Beta_pvalue = 1.0,
        pval = 1.0
      ))
    } else {
      return(data.frame(
        Converged = TRUE,
        Beta = var_summary$coefficients[2,1],
        SE = var_summary$coefficients[2,2],
        Beta_pvalue = var_summary$coefficients[2,4],
        pval = var_summary$coefficients[2,4]
      ))
    }
  } else{
    return(NULL)
  }
}

###Categorical###
regress_cat <- function(data, varying_covariates, outcome, var_name, regression_family){
  # Create a regression formula and a restricted regression formula
  if(length(varying_covariates)>0){
    fmla <- paste(quote_name(outcome), "~", quote_name(var_name), "+", paste(lapply(varying_covariates, quote_name), collapse="+"), sep="")
    fmla_restricted <- paste(quote_name(outcome), "~", paste(lapply(varying_covariates, quote_name), collapse="+"), sep="")
  } else {
    fmla <- paste(quote_name(outcome), "~", quote_name(var_name), sep="")
    fmla_restricted <- paste(quote_name(outcome), "~1", sep="")
  }

  # Run GLM Functions
  var_result <- tryCatch(glm(stats::as.formula(fmla), family=regression_family, data=data, na.action=na.omit),
                         error=function(e) warn_on_e(var_name, e))
  # Only run if the full model did not have an error (isn't NULL)
  if(!is.null(var_result)){
    restricted_result <- tryCatch(glm(stats::as.formula(fmla_restricted), family=regression_family,
                                      data=var_result$model),  # Use the same data as the full model
                                  error=function(e) warn_on_e(var_name, e))
  } else {
    restricted_result <- NULL
  }

  if(!is.null(restricted_result)){
    # Get the LRT using anova
    lrt <- list(p=NA)  # Start with NA for p in case anova fails
    tryCatch(lrt <- anova(var_result, restricted_result, test = "LRT"), error=function(e) warn_on_e(var_name, e))
    result <- data.frame(
      Converged = var_result$converged,
      LRT_pvalue = lrt$`Pr(>Chi)`[2],
      Diff_AIC = var_result$aic - restricted_result$aic,
      pval = lrt$`Pr(>Chi)`[2]
    )
    # Expand to multiple rows if reporting betas
    if(report_categorical_betas){
      vars <- setdiff(names(var_result$coefficients), names(restricted_result$coefficients))
      var_summary <- summary(var_result)
      num_coeff_cols <- length(var_summary$coefficients)/nrow(var_summary$coefficients)
      if (num_coeff_cols < 2){
        # Assume non-convergence if no p values are generated
        return(NULL)
      } else if (num_coeff_cols == 2) {
          Beta = var_summary$coefficients[vars,1]
          SE = var_summary$coefficients[vars,2]
          Beta_pvalue = 1.0
      } else {
          Beta = var_summary$coefficients[vars,1]
          SE = var_summary$coefficients[vars,2]
          Beta_pvalue = var_summary$coefficients[vars,4]
      }
      result <- result[rep(1, length(vars)), ]
      result$Category <- vars
      result$Beta <- Beta
      result$SE <- SE
      result$Beta_pvalue <- Beta_pvalue
    }
    return(result)
  } else {
    return(NULL)
  }
}


regress_cat_survey <- function(data, varying_covariates, outcome, var_name, regression_family,
                               weight_values, strata_values, fpc_values, id_values, subset_array, ...) {
  # Create survey design
  if(is.null(id_values)){
    survey_design <- survey::svydesign(ids = ~1,
                                       weights = weight_values,
                                       data = data,
                                       strata = strata_values,
                                       fpc = fpc_values, ...)
  } else{
    survey_design <- survey::svydesign(ids = id_values,
                                       weights = weight_values,
                                       data = data,
                                       strata = strata_values,
                                       fpc = fpc_values, ...)
  }
  # Subset the survey design.  Including where the variable is NA to ensure same data is used for both models.
  subset_array <- subset_array & !is.na(data[var_name])
  survey_design <- subset(survey_design, subset_array)

  # Create a regression formula and a restricted regression formula
  if(length(varying_covariates)>0){
    fmla <- paste(quote_name(outcome), "~", quote_name(var_name), "+", paste(lapply(varying_covariates, quote_name), collapse="+"), sep="")
    fmla_restricted <- paste(quote_name(outcome), "~", paste(lapply(varying_covariates, quote_name), collapse="+"), sep="")
  } else {
    fmla <- paste(quote_name(outcome), "~", quote_name(var_name), sep="")
    fmla_restricted <- paste(quote_name(outcome), "~1", sep="")
  }

  # Results using surveyglm
  survey_design <<- survey_design  # needed to make the anova function work
  regression_family <<- regression_family  # needed to make the anova function work
  var_result <- tryCatch(survey::svyglm(stats::as.formula(fmla), design=survey_design, family=regression_family, na.action=na.omit),
                         error=function(e) warn_on_e(var_name, e))
  # Restricted result uses the design from the full result to ensure the same observations are used.
  # Otherwise some dropped by 'na.omit' may be included in the restricted model.
  # Only run if the full model did not have an error (isn't NULL)
  if(!is.null(var_result)){
    restricted_result <- tryCatch(survey::svyglm(stats::as.formula(fmla_restricted), design=var_result$survey.design,
                                                 family=regression_family),
                                  error=function(e) warn_on_e(var_name, e))
  } else {
    restricted_result <- NULL
  }

  # Collect results if restricted_result is not NULL
  if(!is.null(restricted_result)){
    # Get the LRT using anova
    lrt <- list(p=NA)  # Start with NA for p in case anova fails
    tryCatch(lrt <- anova(var_result, restricted_result, method = "LRT"), error=function(e) warn_on_e(var_name, e))
    result <- data.frame(
      Converged = var_result$converged,
      LRT_pvalue = lrt$p,
      pval = lrt$p
    )
    # Expand to multiple rows if reporting betas
    if(report_categorical_betas){
      vars <- setdiff(names(var_result$coefficients), names(restricted_result$coefficients))
      var_summary <- summary(var_result)
      result <- result[rep(1, length(vars)), ]
      result$Category <- vars
      result$Beta <- var_summary$coefficients[vars, 1]
      result$SE <- var_summary$coefficients[vars, 2]
      result$Beta_pvalue <- var_summary$coefficients[vars, 4]
    }
    return(result)
  } else {
    return(NULL)
  }
}

# General Regression function which applies some filters/tests before calling the actual regression
regress <- function(data, y, var_name, covariates, min_n, allowed_nonvarying, regression_family, var_type,
                    use_survey, single_weight, weights, strata, fpc, ids, subset_array, drop_unweighted, ...){
  # The result list will be used to update results for this variable
  result = list()

  # Figure out which observations will drop due to NAs
  subset_data <- complete.cases(data[, c(y, var_name, covariates)])  # Returns a boolean array
  if(!is.null(subset_array)){
    subset_data <- subset_data & subset_array
  }

  # Gather survey info if needed
  if(use_survey){
    # Get weight
    if(single_weight){
      weight <- weights
    } else {
      weight <- weights[[var_name]]
    }

    # Record weight name
    if(is.null(weight)){
      warning(paste(var_name, " had a NULL result because no weight was specified"))
      return(NULL)
    } else {
      result$weight <- weight
    }
    # Get weight values, returning early if there is a problem with the weight
    if(!(weight %in% names(data))){
      # Weight values are missing
      warning(paste(var_name, " had a NULL result because its weight (", weight, ") was not found"))
      result$weight <- paste(weight, " (not found)")
      return(data.frame(result, stringsAsFactors = FALSE))
    }
    missing_weight_count <- sum(!is.na(data[var_name]) & is.na(data[weight]) & subset_data)
    if(missing_weight_count > 0){
      # Some weights in the subset are missing when the variable is not
      warning(paste(var_name, " had a NULL result because its weight (", weight, ") had ", missing_weight_count, " missing values when the variable was not missing"))
      result$weight <- paste0(weight, " (", missing_weight_count, " observations are missing weights)")
      if (!drop_unweighted){
        # Return early with no result if dropping unweighted was not enabled
        return(data.frame(result, stringsAsFactors = FALSE))
      } else {
        # Drop rows with missing weights
        subset_data <- subset_data & !(!is.na(data[var_name]) & is.na(data[weight]))
      }
    }
    # Get weights
    weight_values <- data[weight]
    # Fill NA weight values with 0 to pass an internal check by survey
    weight_values[is.na(weight_values),] <- 0

    # Load strata, fpc, and ids
    if(!is.null(strata)){
      strata_values <- data[strata]
    } else {
      strata_values <- NULL
    }
    if(!is.null(fpc)){
      fpc_values <- data[fpc]
    } else {
      fpc_values <- NULL
    }
    if(!is.null(ids)){
      id_values <- data[ids]
    } else {
      id_values <- NULL
    }
  }
  
  # Skip regression if any covariates are constant (after removing NAs) without being specified as allowed
  varying_covariates <- get_varying_covariates(data[subset_data,], covariates, var_name, allowed_nonvarying)
  # If 'get_varying_covarites' returned NULL it found a nonvarying covariate the wasn't allowed)
  if (is.null(varying_covariates) && !is.null(covariates)){
    return(data.frame(result, stringsAsFactors = FALSE))
  }

  # Record N and skip regression if the min_n filter isn't met
  non_na_obs <- sum(subset_data)
  result$N <- non_na_obs
  if (non_na_obs < min_n){
    warning(paste(var_name, " had a NULL result due to the min_n filter (", non_na_obs, " < ", min_n, ")"))
    return(data.frame(result, stringsAsFactors = FALSE))
  }

  # Standardize data if needed
  if(standardize_data){
    allowed_to_scale_cols <- colnames(data) %in% c(y, var_name, varying_covariates)
    numeric_cols <- sapply(data, is.numeric)  # Exclude factors
    binary_cols <- sapply(data, function(s){all(s==0 | s==1 | is.na(s))})  # Exclude binary encoded as 0/1/missing
    scale_cols <- allowed_to_scale_cols & numeric_cols & !binary_cols
    data[scale_cols] <- scale(data[scale_cols])
  }

  # Run Regression for the single variable
  if(!use_survey){
    if(var_type == 'bin'){
      regression_result <- regress_cont(data, varying_covariates, outcome=y, var_name, regression_family)
    } else if(var_type == 'cat'){
      regression_result <- regress_cat(data, varying_covariates, outcome=y, var_name, regression_family)
    } else if(var_type == 'cont'){
      regression_result <- regress_cont(data, varying_covariates, outcome=y, var_name, regression_family)
    }
  } else {
    if(var_type == 'bin'){
      regression_result <- regress_cont_survey(data, varying_covariates, outcome=y, var_name, regression_family,
                                               weight_values, strata_values, fpc_values, id_values,
                                               subset_data, ...)
    } else if(var_type == 'cat'){
      regression_result <- regress_cat_survey(data, varying_covariates, outcome=y, var_name, regression_family,
                                              weight_values, strata_values, fpc_values, id_values,
                                              subset_data, ...)
    } else if(var_type == 'cont'){
      regression_result <- regress_cont_survey(data, varying_covariates, outcome=y, var_name, regression_family,
                                               weight_values, strata_values, fpc_values, id_values,
                                               subset_data, ...)
    }
  }
  # Update result with the regression results
  if(!is.null(regression_result)){
    regression_result[names(result)] <- result
  } else {
    regression_result <- data.frame(result, stringsAsFactors = FALSE)
  }

  # Return
  return(regression_result)

}


#' ewas
#'
#' Run environment-wide association study, optionally using \code{\link[survey]{svydesign}} from the \pkg{survey} package
#' Note: It is possible to specify \emph{ids} and/or \emph{strata}.  When \emph{ids} is specified without \emph{strata},
#' the standard error is infinite and the anova calculation for categorical variables fails.  This is due to the
#' \href{http://r-survey.r-forge.r-project.org/survey/exmample-lonely.html}{lonely psu} problem.
#' @param d data.frame containing all of the data
#' @param bin_vars List of variables to regress that are binary
#' @param cat_vars List of variables to regress that are categorical
#' @param cont_vars  List of variables to regress that are continuous
#' @param y name(s) of response variable(s)
#' @param bin_covars List of covariates that are continuous
#' @param cat_covars List of covariates that are categorical
#' @param cont_covars List of covariates that are continuous
#' @param regression_family family for the regression model as specified in glm ('gaussian' by default)
#' @param allowed_nonvarying list of covariates that are excluded from the regression when they do not vary instead of returning a NULL result.
#' @param min_n minimum number of observations required (after dropping those with NA values) before running the regression (200 by default)
#' @param weights NULL by default (for unweighted).  May be set to a string name of a single weight to use for every variable, or a named list that maps variable names to the weights that should be used for that variable's regression
#' @param ids NULL by default (for no clusters).  May be set to a string name of a column in the data which provides cluster IDs.
#' @param strata NULL by default (for no strata).  May be set to a string name of a column in the data which provides strata IDs.
#' @param fpc NULL by default (for no fpc).  May be set to a string name of a column in the data which provides fpc values.
#' @param subset_array NULL by default (for no subset).  May be set to a boolean array used to subset the data after creating the design
#' @param report_categorical_betas FALSE by default
#' @param standardize_data FALSE by default
#' @param ... other arguments passed to svydesign which are ignored if 'weights' is NULL
#' @return data frame containing following fields Variable, Sample Size, Converged, SE, Beta, Variable p-value, LRT, AIC, pval, outcome, weight
#' @export
#' @family analysis functions
#' @examples
#' \dontrun{
#' ewas(d, cat_vars, cont_vars, y, cat_covars, cont_covars, regression_family)
#' }
ewas <- function(d, bin_vars=NULL, cat_vars=NULL, cont_vars=NULL, y,
                 bin_covars=NULL, cat_covars=NULL, cont_covars=NULL,
                 regression_family="gaussian", allowed_nonvarying=NULL, min_n=200, weights=NULL,
                 ids=NULL, strata=NULL, fpc=NULL, subset_array=NULL,
                 report_categorical_betas=FALSE, standardize_data=FALSE, ...){
  # Record start time
  t1 <- Sys.time()

  # Record global options
  report_categorical_betas <<- report_categorical_betas
  standardize_data <<- standardize_data

  # Validate inputs
  #################
  if(missing(y)){
    stop("Please specify an outcome 'y' variable")
  }
  if(is.null(bin_vars)){
    bin_vars <- list()
  }
  if(is.null(cat_vars)){
    cat_vars <- list()
  }
  if(is.null(cont_vars)){
    cont_vars <- list()
  }
  if(is.null(bin_covars)){
    bin_covars <- list()
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
  if(!is.null(ids)){
    if(!(ids %in% colnames(d))){
      stop(paste("'ids' was specified (", ids, ") but not found in the data", sep=""))
    }
  }
  if(!is.null(strata)){
    if(!(strata %in% colnames(d))){
      stop(paste("'strata' was specified (", strata, ") but not found in the data", sep=""))
    }
  }
  if(!is.null(fpc)){
    if(!(fpc %in% colnames(d))){
      stop(paste("'fpc' was specified (", fpc, ") but not found in the data", sep=""))
    }
  }
  if(!is.null(ids) && is.null(strata) && is.null(fpc)){
    warning("PSU IDs were specified without strata or fpc, preventing calculation of standard error")
  }

  # Ignore the covariates, outcome, and ID if they were included in the variable lists
  remove <- c(y, bin_covars, cat_covars, cont_covars, "ID")
  bin_vars <- setdiff(bin_vars, remove)
  cat_vars <- setdiff(cat_vars, remove)
  cont_vars <- setdiff(cont_vars, remove)
  # Ignore the outcome, and ID if they were included in the covariates lists
  remove <- c(y, "ID")
  bin_covars <- setdiff(bin_covars, remove)
  cat_covars <- setdiff(cat_covars, remove)
  cont_covars <- setdiff(cont_covars, remove)

  # Check data
  if(class(d)[1] != "data.frame"){
    stop("Data must be a data.frame object")
  }

  # Check weights
  if(is.null(weights)){
    print("Running without a survey design adjustment")
    use_survey <- FALSE
  } else if(class(weights) == "character"){
    single_weight <- TRUE
    if(!(weights %in% names(d))){
      stop(paste(weights, "was specified as the weight, but was not found in the dataframe", sep=" "))
    }
    print("Running with a single weight used for all variables")
    use_survey <- TRUE
  } else if(class(weights) == "list"){
    single_weight <- FALSE
    print("Running with specific weights assigned for each variable")
    use_survey <- TRUE
  } else {
    stop("weights must be a string or a list")
  }

  #Correct the types and check for IDs
  #####################################
  # ID
  if(is.element('ID', names(d))==FALSE){stop("Please add ID to the data as column 1")}
  d$ID <- factor(d$ID)
  # Binary
  if(length(bin_vars) > 0){d[bin_vars] <- lapply(d[bin_vars], factor)}
  if(length(bin_covars) > 0){d[bin_covars] <- lapply(d[bin_covars], factor)}
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

  # Get a combined vector of covariates (must 'unlist' lists to vectors)
  covariates <- c(unlist(bin_covars), unlist(cat_covars), unlist(cont_covars))

  # Run Regressions
  #################

  # Create a placeholder dataframe for results, anything not updated will be NA
  if(report_categorical_betas & length(cat_vars)>0){
    n <- length(bin_vars) + length(cont_vars) + sum(sapply(cat_vars, function(v) length(table(d[v]))-1))
  } else {
    n <- length(bin_vars) + length(cat_vars) + length(cont_vars)
  }
  ewas_result_df <- data.frame(Variable = character(n),
                               Variable_type = character(n),
                               N = numeric(n),
                               Converged = logical(n),
                               Beta = numeric(n),
                               SE = numeric(n),
                               Beta_pvalue = numeric(n),
                               LRT_pvalue = numeric(n),
                               Diff_AIC = numeric(n),
                               pval = numeric(n),
                               outcome = character(n),
                               weight = character(n),
                               stringsAsFactors = FALSE)
  ewas_result_df[] <- NA  # Fill df with NA values
  ewas_result_df$Converged <- FALSE  # Default to not converged
  i <- 0 # Increment before processing each variable

  # Process binary variables, if any
  print(paste("Processing ", length(bin_vars), " binary variables", sep=""))
  for(var_name in bin_vars){
    # Update var name and outcome
    i <- i + 1
    ewas_result_df$Variable[i] <- var_name
    ewas_result_df$outcome[i] <- y
    ewas_result_df$Variable_type[i] <- "binary"

    result <- regress(d, y, var_name, covariates, min_n, allowed_nonvarying, regression_family, var_type="bin",
                      use_survey, single_weight, weights, strata, fpc, ids, subset_array, ...)

    # Save results
    if(!is.null(result)){
      ewas_result_df[i, colnames(result)] <- result
    }
  }

  # Process categorical variables, if any
  print(paste("Processing ", length(cat_vars), " categorical variables", sep=""))
  for(var_name in cat_vars){
    result <- regress(d, y, var_name, covariates, min_n, allowed_nonvarying, regression_family, var_type="cat",
                      use_survey, single_weight, weights, strata, fpc, ids, subset_array, ...)

    # Save results
    if(!is.null(result)){
      for(ridx in 1:nrow(result)){
        i <- i + 1
        # Update var name and outcome
        ewas_result_df$Variable[i] <- var_name
        ewas_result_df$outcome[i] <- y
        ewas_result_df$Variable_type[i] <- "categorical"
        # Update results
        ewas_result_df[i, colnames(result)] <- result[ridx,]
      }
    }
  }

  # Process continuous variables, if any
  print(paste("Processing ", length(cont_vars), " continuous variables", sep=""))
  for(var_name in cont_vars){
    # Update var name and outcome
    i <- i + 1
    ewas_result_df$Variable[i] <- var_name
    ewas_result_df$outcome[i] <- y
    ewas_result_df$Variable_type[i] <- "continuous"

    result <- regress(d, y, var_name, covariates, min_n, allowed_nonvarying, regression_family, var_type="cont",
                      use_survey, single_weight, weights, strata, fpc, ids, subset_array, ...)

    # Save results
    if(!is.null(result)){
      ewas_result_df[i, colnames(result)] <- result
    }
  }

  t2 <- Sys.time()
  print(paste("Finished in", round(as.numeric(difftime(t2,t1, units="secs")), 6), "secs", sep=" "))
  n_null_results <- sum(is.null(ewas_result_df$pval))
  if (n_null_results > 0){
    warning(paste(n_null_results, "of", nrow(ewas_result_df), "variables had a NULL result due to an error (see earlier warnings for details)"))
  }

  # Sort by pval
  # TODO: Sort differently
  if(report_categorical_betas){
    ewas_result_df <- ewas_result_df[order(ewas_result_df$pval, ewas_result_df$Beta_pvalue),]
  } else {
    ewas_result_df <- ewas_result_df[order(ewas_result_df$pval),]
  }

  # Replace NA with 'None' for correct conversion back to Pandas format
  ewas_result_df[is.na(ewas_result_df)]='None'

  return(ewas_result_df)
}
