# This script loads several datasets from the survey library and saves them
# for testing in both R and Python

set.seed(1855)

library(devtools)
if (!require('survey')) {
  install.packages('survey', repos = "http://cran.us.r-project.org")
}
library('survey')

# Change to the output folder
current_dir <- getwd()
output_dir <- file.path(current_dir, "test_data_files")
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

#nostrat
fpc_nostrat <- fpc
fpc_nostrat$Nh <- 30
write.csv(fpc_nostrat, 'fpc_nostrat_data.csv', row.names=FALSE)

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
nhanes$HI_CHOL <- as.factor(nhanes$HI_CHOL)
nhanes$race <- as.factor(nhanes$race)
nhanes$agecat <- as.factor(nhanes$agecat)
nhanes$RIAGENDR <- as.factor(nhanes$RIAGENDR)
write.csv(nhanes, 'nhanes_data.csv', row.names=FALSE)

# Save a copy with a random normal variable for testing subsets
nhanes$subset <- rnorm(nrow(nhanes))
write.csv(nhanes, 'nhanes_data_subset.csv', row.names=FALSE)
nhanes[ , !(names(nhanes) == "subset")] # Delete column after saving

# Save a copy with NAs
nhanes_NAs <- nhanes
nhanes_NAs$race[2:800] <- NA
write.csv(nhanes_NAs, 'nhanes_NAs_data.csv', row.names=FALSE)

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