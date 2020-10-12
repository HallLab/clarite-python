"""
Regression Classes
==================

clarite.analyze.ewas
--------------------

The `regression_kind` parameter can be set to use one of three regression classes:

.. autoclass:: GLMRegression

.. autoclass:: WeightedGLMRegression

.. autoclass:: RSurveyRegression

Alternatively, a custom subclass of `Regression` can be created and used:

.. autoclass:: Regression


clarite.analyze.interactions
----------------------------

.. autoclass:: InteractionRegression

"""

from .glm_regression import GLMRegression
from .r_survey_regression import RSurveyRegression
from .weighted_glm_regression import WeightedGLMRegression
from .base import Regression

from.interaction_regression import InteractionRegression

__all__ = [GLMRegression, RSurveyRegression, WeightedGLMRegression, Regression,
           InteractionRegression]
