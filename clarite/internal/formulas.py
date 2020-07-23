import re
import pandas as pd

subset_formula_re = re.compile("(?P<variable>.+)(?P<operator><|<=|==|!=|>=|>)(?P<value>.+)")


class SubsetFormula:
    def __init__(self, formula_str: str):
        # Parse string
        match = subset_formula_re.match(formula_str)
        if match is None:
            raise ValueError(f"Could not parse '{formula_str}' into a formula")
        else:
            params = match.groupdict()

        self.variable = params['variable'].strip()
        self.operator = params['operator']
        self.value = params['value'].strip()

        # Validate Parameters
        valid_operators = {'<', '<=', '==', '!=', '>=', '>'}
        if self.operator not in valid_operators:
            raise ValueError(f"Parsed '{formula_str}', but '{self.operator}' is not a valid operator")
        if self.operator not in {'==', '!='} and not self.value.isnumeric():
            raise ValueError(f"Parsed '{formula_str}', but the value ({self.value}) isn't numeric"
                             f" and the operator ({self.operator}) requires it to be.")

    def __str__(self):
        return f"{self.variable} {self.operator} {self.value}"

    def get_bool_filter(self, data: pd.DataFrame):
        if self.variable not in list(data):
            raise ValueError(f"The variable ({self.variable}) was not found in the DataFrame")

        # Create a boolean filter by applying the subset
        if self.operator == '<':
            bool_filter = data[self.variable] < self.value
        elif self.operator == "<=":
            bool_filter = data[self.variable] <= self.value
        elif self.operator == "==":
            bool_filter = data[self.variable] == self.value
        elif self.operator == "!=":
            bool_filter = data[self.variable] != self.value
        elif self.operator == ">=":
            bool_filter = data[self.variable] >= self.value
        elif self.operator == ">":
            bool_filter = data[self.variable] > self.value
        else:
            raise ValueError(f"Unknown operator in formula: {str(self)}")

        return bool_filter
