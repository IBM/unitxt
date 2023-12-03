import unittest
from dataclasses import field

from src.unitxt.dataclass import (
    AbstractField,
    AbstractFieldError,
    Dataclass,
    FinalField,
    FinalFieldError,
    NonPositionalField,
    RequiredField,
    RequiredFieldError,
    UnexpectedArgumentError,
    fields,
    fields_names,
)


class TestDataclass(unittest.TestCase):
    def test_dataclass(self):
        class GrandParent(Dataclass):
            a: int
            b: int

        class Parent(GrandParent):
            a = 1

        class Child(Parent):
            c: int

        child = Child(b=2, c=3)

        with self.subTest("test_parent"):
            self.assertEqual(child.a, 1)

        self.assertEqual(child.a, 1)
        self.assertEqual(child.b, 2)
        self.assertEqual(child.c, 3)

    def test_final_field(self):
        class GrandParent(Dataclass):
            a: int = 3
            b: int
            t: list = field(default_factory=list)

            def work(self):
                return self.a + self.b

        class Parent(GrandParent):
            a: int = FinalField(default=1)

        with self.assertRaises(FinalFieldError):

            class Child(Parent):
                a: int = 2

    def test_required_field_one_generation(self):
        class Dummy(Dataclass):
            a: int

        with self.subTest("simple test"):
            with self.assertRaises(RequiredFieldError):
                Dummy()

        with self.subTest("test with input value"):
            d = Dummy(a=1)
            self.assertEqual(d.a, 1)

        with self.subTest("test with implicit decleration"):

            class Dummy(Dataclass):
                a: float = RequiredField()

        with self.assertRaises(RequiredFieldError):
            Dummy()

    def test_abstract_field_one_generation(self):
        class Dummy(Dataclass):
            a: int = AbstractField()

        with self.assertRaises(AbstractFieldError):
            Dummy()

        with self.assertRaises(AbstractFieldError):
            Dummy(a=1)

    def test_abstract_field(self):
        class GrandParent(Dataclass):
            a: int = AbstractField()
            b: int = AbstractField()

        class Parent(GrandParent):
            # add missing field
            a: float = AbstractField()

        with self.assertRaises(AbstractFieldError):
            Parent(b=2)

        with self.assertRaises(AbstractFieldError):
            Parent(a=2, b=2)

        class Child(Parent):
            a: int
            b: int

        with self.assertRaises(RequiredFieldError):
            c = Child(a=2)

        c = Child(a=2, b=2)
        self.assertEqual(c.a, 2)
        self.assertEqual(c.b, 2)

    def test_multiple_inheritance(self):
        class GrandParent(Dataclass):
            a: int = 0

        class Parent1(GrandParent):
            a: int = 1

        class Parent2(GrandParent):
            b: int = 2

        class Child(Parent1, Parent2):
            c: int

        child = Child(c=3)

        with self.subTest("test_parent"):
            self.assertEqual(child.a, 1)
            self.assertEqual(child.b, 2)
            self.assertEqual(child.c, 3)

    def test_final_with_multiple_inheritance(self):
        class GrandParent(Dataclass):
            a: int = 0

        class Parent1(GrandParent):
            a: int = FinalField(default=1)

        class Parent2(GrandParent):
            b: int = 2

        with self.assertRaises(FinalFieldError):

            class Child(Parent1, Parent2):
                a: int = 2
                c: int

        class Child(Parent1, Parent2):
            c: int

        child = Child(c=3)

        with self.subTest("test_parent"):
            self.assertEqual(child.a, 1)
            self.assertEqual(child.b, 2)
            self.assertEqual(child.c, 3)

    def test_abstract_with_multiple_inheritance(self):
        class GrandParent(Dataclass):
            a: int = 0

        class Parent1(GrandParent):
            a: int = AbstractField()

        class Parent2(GrandParent):
            b: int = 2

        with self.assertRaises(AbstractFieldError):
            Parent1(a=1)

        class Child(Parent1, Parent2):
            a = 1
            c: int

        self.assertListEqual(fields_names(Child), ["a", "b", "c"])

        child = Child(b=2, c=3)

        with self.subTest("test_parent"):
            self.assertEqual(child.a, 1)
            self.assertEqual(child.b, 2)
            self.assertEqual(child.c, 3)

    def test_no_collision(self):
        class GrandParent(Dataclass):
            a: int = 0

        class Parent1(GrandParent):
            a: int = 1

        class Parent2(GrandParent):
            a: int = 2

        class Child(Parent1, Parent2):
            c: int

        gp = GrandParent()
        p1 = Parent1()
        p2 = Parent2()
        p3 = Parent1()
        c1 = Child(c=0)
        c2 = Child(c=1)

        self.assertEqual(gp.a, 0)
        self.assertEqual(p1.a, 1)
        self.assertEqual(p2.a, 2)
        self.assertEqual(p3.a, 1)
        self.assertEqual(c1.c, 0)
        self.assertEqual(c2.c, 1)

    def test_no_backward_affects(self):
        class Parent(Dataclass):
            a: int = 1

        class Chiled(Parent):
            a: int = 2
            b: int = 3

        c = Chiled()
        p = Parent()
        parent_fields = fields(Parent)
        self.assertEqual(len(parent_fields), 1)
        self.assertEqual(parent_fields[0].name, "a")
        self.assertEqual(p.a, 1)
        self.assertEqual(c.a, 2)

    def test_filling_requirement_with_mixin(self):
        class GrandParent(Dataclass):
            a: int = 0

        class Parent1(GrandParent):
            b: int = 2

        class Mixin(Dataclass):
            a: int = 2

        class Child(Mixin, Parent1):
            c: int

        child = Child(b=2, c=3)

        self.assertEqual(child.a, 2)
        self.assertEqual(child.b, 2)
        self.assertEqual(child.c, 3)

    def test_raising_unexpected_keyword_argument_error(self):
        class Dummy(Dataclass):
            b = 1  # not a field!!!
            a: int = 1

        self.assertListEqual(fields_names(Dummy), ["a"])

        with self.assertRaises(UnexpectedArgumentError):
            Dummy(b=2)

    def test_non_positional_fields(self):
        class Dummy(Dataclass):
            a: int = NonPositionalField()

        with self.assertRaises(UnexpectedArgumentError):
            Dummy(1)

        d = Dummy(a=1)
        self.assertEqual(d.a, 1)

        class Dummy(Dataclass):
            a: int = NonPositionalField()
            b: int

        with self.assertRaises(UnexpectedArgumentError):
            Dummy(1, 2)

        with self.assertRaises(TypeError):
            Dummy(1, b=2)  # both assigning to b

        d = Dummy(2, a=1)

        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 2)

    def test_unexpected_arguments_saving(self):
        class Dummy(Dataclass):
            __allow_unexpected_arguments__ = True
            a: int = 1

        d = Dummy(1, 2, c=3)
        self.assertEqual(d.a, 1)
        self.assertTupleEqual(d._argv, (2,))
        self.assertDictEqual(d._kwargs, {"c": 3})
