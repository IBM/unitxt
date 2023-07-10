from .operator import SourceSequntialOperator


class Recipe:
    pass


class SequentialRecipe(Recipe, SourceSequntialOperator):
    pass


# class CommonRecipe(Recipe, SourceOperator):

#     task_card: TaskCard = None

#     def prepare(self):
#         self.recipe = SequentialRecipe(
#             self.task_card.preprocess,
#             self.task_card.task,
#             RenderTemplatedICL(

#             )
#         )
