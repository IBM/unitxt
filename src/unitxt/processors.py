from .operator import BaseFieldOperator


class ToString(BaseFieldOperator):
    def process(self, instance):
        return str(instance)


# add_to_catalog(ToString('prediction'), 'processors', 'to_string')
