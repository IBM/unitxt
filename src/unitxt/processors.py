from .operator import FieldOperator


class ToString(FieldOperator):
    def process(self, instance):
        return str(instance)


# add_to_catalog(ToString('prediction'), 'processors', 'to_string')
