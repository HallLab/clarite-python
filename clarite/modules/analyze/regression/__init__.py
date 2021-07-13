"""
Regression Classes
==================

Base Class
----------

.. autoclass:: Regression


clarite.analyze.association_study
---------------------------------

The `regression_kind` parameter can be set to use one of three regression classes, or a custom subclass of `Regression`
can be created.

.. autoclass:: GLMRegression

.. autoclass:: WeightedGLMRegression

.. autoclass:: RSurveyRegression


clarite.analyze.interaction_study
---------------------------------

.. autoclass:: InteractionRegression

"""

from .glm_regression import GLMRegression
from .r_survey_regression import RSurveyRegression
from .weighted_glm_regression import WeightedGLMRegression
from .base import Regression

from .interaction_regression import InteractionRegression

builtin_regression_kinds = {
    "glm": GLMRegression,
    "weighted_glm": WeightedGLMRegression,
    "r_survey": RSurveyRegression,
}


__all__ = [
    "GLMRegression",
    "RSurveyRegression",
    "WeightedGLMRegression",
    "Regression",
    "InteractionRegression",
    "builtin_regression_kinds",
]
