# Let's modify the code to allow the finalfield and field functions to accept the same parameters as dataclasses.field
class AbstractFieldValue:
    def __init__(self):
        raise TypeError("Abstract field must be overridden in subclass")

import dataclasses

class FinalField:
    def __init__(self, *, default=dataclasses.MISSING, default_factory=dataclasses.MISSING,
                 init=True, repr=True, hash=None, compare=True, metadata=None):
        self.field = dataclasses.field(default=default, default_factory=default_factory,
                                       init=init, repr=repr, hash=hash, compare=compare, metadata=metadata)

def abstractfield():
    return dataclasses.field(default_factory=AbstractFieldValue)

def finalfield(*, default=dataclasses.MISSING, default_factory=dataclasses.MISSING,
               init=True, repr=True, hash=None, compare=True, metadata=None):
    return FinalField(default=default, default_factory=default_factory,
                      init=init, repr=repr, hash=hash, compare=compare, metadata=metadata)

def field(*, default=dataclasses.MISSING, default_factory=dataclasses.MISSING,
          init=True, repr=True, hash=None, compare=True, metadata=None):
    return dataclasses.field(default=default, default_factory=default_factory,
                             init=init, repr=repr, hash=hash, compare=compare, metadata=metadata)

class DataclassMeta(type):
    def __new__(cls, name, bases, attrs):
        attrs['__finalfields__'] = attrs.get('__finalfields__', [])
        for base in bases:
            if issubclass(base, Dataclass) and hasattr(base, '__finalfields__'):
                for field in base.__finalfields__:
                    if field in attrs:
                        raise TypeError(f"Final field '{field}' cannot be overridden in subclass")
                    attrs['__finalfields__'].append(field)

        for attr_name, attr_value in list(attrs.items()):
            if isinstance(attr_value, FinalField):
                attrs[attr_name] = attr_value.field  # Replace the final field marker with the actual field
                attrs['__finalfields__'].append(attr_name)

        new_class = super().__new__(cls, name, bases, attrs)
        new_class = dataclasses.dataclass(new_class)

        return new_class



class Dataclass(metaclass=DataclassMeta):
    pass

if __name__ == '__main__':
# Test classes
    class GrandparentClass(Dataclass):
        abstract_field: int = abstractfield()
        final_field: str = finalfield(default_factory=lambda: 'Hello')

    class ParentClass(GrandparentClass):
        pass

    try:
        class CorrectChildClass(ParentClass):
            abstract_field: int = 1  # This correctly overrides the abstract field
        correct_child_class_instance = CorrectChildClass()
        print(f'CorrectChildClass instance: {correct_child_class_instance} - passed')
    except Exception as e:
        print(f'CorrectChildClass: {str(e)} - failed')

    try:
        class IncorrectChildClass1(ParentClass):
            pass  # This fails to override the abstract field
        print(f'IncorrectChildClass1: {IncorrectChildClass1} - passed')
    except Exception as e:
        print(f'IncorrectChildClass1: {str(e)} - failed')

    try:
        incorrect_child_class1_instance = IncorrectChildClass1()
        print(f'IncorrectChildClass1 instance: {incorrect_child_class1_instance} - failed')
    except Exception as e:
        print(f'IncorrectChildClass1 instantiation: {str(e)} - passed')

    # Testing the final field functionality

    try:
        class IncorrectChildClass2(ParentClass):
            final_field: str = 'Hello'  # This attempts to override the final field
        print(f'IncorrectChildClass2: {IncorrectChildClass2} - failed')
    except Exception as e:
        print(f'IncorrectChildClass2: {str(e)} - passed')
