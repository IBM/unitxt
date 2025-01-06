from unitxt import add_to_catalog
from unitxt.serializers import SQLSchemaSerializer

add_to_catalog(SQLSchemaSerializer(), "serializers.text2sql.schema", overwrite=True)
