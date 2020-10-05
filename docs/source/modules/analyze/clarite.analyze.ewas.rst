clarite.analyze.ewas
====================

.. currentmodule:: clarite.analyze

.. autofunction:: ewas

Available Regression Classes
----------------------------

The `regression_kind` parameter can be set to use one of these three regression classes, or a custom subclass
of `clarite.internal.regression.Regression` can be created and used.

.. autoclass:: clarite.internal.regression.GLMRegression

.. autoclass:: clarite.internal.regression.WeightedGLMRegression

.. autoclass:: clarite.internal.regression.RSurveyRegression