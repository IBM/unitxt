from unitxt import add_to_catalog
from unitxt.serializers import SQLDatabaseAsSchemaSerializer

add_to_catalog(
    SQLDatabaseAsSchemaSerializer(), "serializers.text2sql.schema", overwrite=True
)
