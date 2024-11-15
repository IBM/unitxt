
def get_parsed_context(context: dict[str,str]):
     return '\n'.join([f"{key}: {value}" for key, value in context.items()]) if len(context) > 1 or len(context) == 0 or list(context.keys())[0].lower() != 'context' else context[list(context.keys())[0]]
