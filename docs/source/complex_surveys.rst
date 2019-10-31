===================
Complex Survey Data
===================

CLARITE provides preliminary support for handling complex survey designs, similar to how the r-package *survey* works.

A SurveyDesignSpec can be created, which is used to obtain survey design objects for specific variables:

.. code-block:: python

    sd_discovery = clarite.survey.SurveyDesignSpec(survey_df=survey_design_discovery,
                                                   strata="SDMVSTRA",
                                                   cluster="SDMVPSU",
                                                   nest=True,
                                                   weights=weights_discovery,
                                                   single_cluster='scaled')


There are a few different options for the 'single_cluster' parameter, which controls how strata with single clusters are handled in the linearized covariance calculation:
    
    * *error* - Throw an error
    * *scaled* - Use the average value of other strata
    * *centered* - Use the average of all observations
    * *certainty* - Single-cluster strata don't contribute to the variance

After a SurveyDesignSpec is created, it can be passed into the ewas function to utilize the survey design parameters:

.. code-block:: python

    ewas_discovery = clarite.analyze.ewas("logBMI", covariates, nhanes_discovery_bin, nhanes_discovery_cat, nhanes_discovery_cont, sd_discovery, cov_method='stata')

