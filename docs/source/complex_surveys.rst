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
                                                   single_cluster='adjust',
                                                   drop_unweighted=False)


There are a few different options for the 'single_cluster' parameter, which controls how strata with single clusters are handled in the linearized covariance calculation:
    
    * *fail* - Throw an error (default)
    * *adjust* - Use the average value of all observaitons (conservative)
    * *average* - Use the average of other strata
    * *certainty* - Single-cluster strata don't contribute to the variance

The `drop_unweighted` parameter is False by default- any variables with missing weights will return a missing result.  Setting it to True will simply drop those observations (which may not be strictly correct).

After a SurveyDesignSpec is created, it can be passed into an ewas function to utilize the survey design parameters:

.. code-block:: python

    ewas_discovery = clarite.analyze.ewas(phenotype="logBMI",
                                          covariates=covariates,
                                          data=nhanes_discovery,
                                          survey_design_spec=sd_discovery)

