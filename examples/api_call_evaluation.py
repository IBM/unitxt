import json
from typing import Dict, List, Tuple

from unitxt import get_logger
from unitxt.api import create_dataset, evaluate
from unitxt.inference import (
    CrossProviderInferenceEngine,
)
from unitxt.operators import FieldOperator
from unitxt.processors import PostProcess
from unitxt.task import Task
from unitxt.templates import InputOutputTemplate

logger = get_logger()


# from https://learn.openapis.org/examples/v3.0/petstore-expanded.html
api_spec = """
openapi: 3.0.0
info:
  version: 1.0.0
  title: Swagger Petstore
  description: A sample API that uses a petstore as an example to demonstrate features in the OpenAPI 3.0 specification
  termsOfService: http://swagger.io/terms/
  contact:
    name: Swagger API Team
    email: apiteam@swagger.io
    url: http://swagger.io
  license:
    name: Apache 2.0
    url: https://www.apache.org/licenses/LICENSE-2.0.html
servers:
  - url: https://petstore.swagger.io/v2
paths:
  /pets:
    get:
      description: |
        Returns all pets from the system that the user has access to
      operationId: findPets
      parameters:
        - name: tags
          in: query
          description: tags to filter by
          required: false
          style: form
          schema:
            type: array
            items:
              type: string
        - name: limit
          in: query
          description: maximum number of results to return
          required: false
          schema:
            type: integer
            format: int32
      responses:
        '200':
          description: pet response
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Pet'
        default:
          description: unexpected error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
    post:
      description: Creates a new pet in the store. Duplicates are allowed
      operationId: addPet
      requestBody:
        description: Pet to add to the store
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/NewPet'
      responses:
        '200':
          description: pet response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Pet'
        default:
          description: unexpected error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
  /pets/{{id}}:
    get:
      description: Returns a user based on a single ID, if the user does not have access to the pet
      operationId: find pet by id
      parameters:
        - name: id
          in: path
          description: ID of pet to fetch
          required: true
          schema:
            type: integer
            format: int64
      responses:
        '200':
          description: pet response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Pet'
        default:
          description: unexpected error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
    delete:
      description: deletes a single pet based on the ID supplied
      operationId: deletePet
      parameters:
        - name: id
          in: path
          description: ID of pet to delete
          required: true
          schema:
            type: integer
            format: int64
      responses:
        '204':
          description: pet deleted
        default:
          description: unexpected error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
components:
  schemas:
    Pet:
      allOf:
        - $ref: '#/components/schemas/NewPet'
        - type: object
          required:
            - id
          properties:
            id:
              type: integer
              format: int64
    NewPet:
      type: object
      required:
        - name
      properties:
        name:
          type: string
        tag:
          type: string
    Error:
      type: object
      required:
        - code
        - message
      properties:
        code:
          type: integer
          format: int32
        message:
          type: string
"""

test_set = [
    {
        "user_request": "List 5 pets from the store with tag dogs",
        "reference_query": "curl -X GET 'https://petstore.swagger.io/v2/pets?tags=dogs&limit=5'",
        "api_spec": api_spec,
    },
    {
        "user_request": "Create a pet entry with name Rexy and tag dog. ",
        "reference_query": 'curl -X POST -H "Content-Type: application/json" -d \'{"name": "Rexy", "tag": "dog"}\' https://petstore.swagger.io/v2/pets',
        "api_spec": api_spec,
    },
    {
        "user_request": "Delete pet with id 4 ",
        "reference_query": "curl -X DELETE 'https://petstore.swagger.io/v2/pets/4'",
        "api_spec": api_spec,
    },
    {
        "user_request": "Get pet with id 3 ",
        "reference_query": "curl -X GET 'https://petstore.swagger.io/v2/pets/3'",
        "api_spec": api_spec,
    },
]


class CurlStrToListOfKeyValuePairs(FieldOperator):
    """This class takes a string query api and splits it into a list of key value components.

    These components can then be checked individually (e.g ignoring order)"
    For example:

    curl -X GET -H "Content-Type: application/json" 'https://petstore.swagger.io/v2/pets?tags=dogs&limit=5'

    becomes

    { 'url' : 'curl -X GET -H "Content-Type: application/json" https://petstore.swagger.io/v2/pets', 'tags' : 'dogs', 'limit' : '5'}

    """

    def process_value(self, text: str) -> List[Tuple[str, str]]:
        import re

        text = text.replace("%20", " ")
        text = text.replace("'", "")

        splits = text.split("?")
        split_command = re.split(r"((?=GET|POST|DELETE)GET|POST|DELETE)", splits[0])
        result = {
            "command": split_command[0],
            "operation": split_command[1],
            "url": split_command[2],
        }
        if len(splits) == 1:
            return result
        params = splits[1]
        params_splits = params.split("&")
        for param in params_splits:
            key_value_splits = param.split("=")
            if len(key_value_splits) != 2:
                print(f"Unable to parse key value pair from string {param} in {text}")
                continue
            (key, value) = key_value_splits
            value_splits = value.split(",")
            if len(value_splits) == 1:
                result[f"query_param_{key}"] = f"{value}"

        return result


template = InputOutputTemplate(
    instruction="Generate the API query corresponding to the user request based on the following api specification. Answer only as a CURL command, without any explanations. \n{api_spec}.",
    input_format="{user_request}",
    output_format="{reference_query}",
    postprocessors=[PostProcess(CurlStrToListOfKeyValuePairs())],
)

task = Task(
    input_fields={"user_request": str, "api_spec": str},
    reference_fields={"reference_query": str},
    prediction_type=Dict[str, str],
    metrics=[
        "metrics.key_value_extraction.accuracy",
        "metrics.key_value_extraction.token_overlap",
    ],
)

dataset = create_dataset(
    task=task,
    template=template,
    test_set=test_set,
    split="test",
    demos_pool_size=2,
    num_demos=1,
    demos_taken_from="test",
    demos_removed_from_data=False,
    format="formats.chat_api",
)

model = CrossProviderInferenceEngine(model="llama-3-3-70b-instruct", provider="watsonx")

predictions = model(dataset)
results = evaluate(predictions=predictions, data=dataset)

print("Example prompt:")

print(json.dumps(results.instance_scores[0]["source"], indent=4).replace("\\n", "\n"))

print("Instance Results:")
df = results.instance_scores.to_df(
    columns=[
        "user_request",
        "reference_query",
        "prediction",
        "processed_references",
        "processed_prediction",
        "score",
    ]
)

for index, row in df.iterrows():
    print(f"Row {index}:")
    for col_name, value in row.items():
        print(f"{col_name}: {value}")
    print("-" * 20)


print("Global Results:")
print(results.global_scores.summary)
