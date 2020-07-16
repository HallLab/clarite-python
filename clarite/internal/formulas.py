import re

subset_formula_re = re.compile("(?P<variable>.+)(?P<operator><|<=|==|!=|>=|>)(?P<value>.+)")


class SubsetFormula:
    def __init__(self, formula_str):
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
