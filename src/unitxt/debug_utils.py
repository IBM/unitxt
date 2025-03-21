import inspect


def insert_breakpoint(func):
    # Retrieve the source code of the function
    source_lines, starting_line_no = inspect.getsourcelines(func)

    # Determine the indentation level of the function definition
    indent = len(source_lines[0]) - len(source_lines[0].lstrip())

    # Create the new function source with a breakpoint at the beginning
    new_source_lines = []
    for line in source_lines:
        new_source_lines.append(line[indent:])
        if line.lstrip().startswith("def "):
            # Insert a blank line and the breakpoint after the function definition
            new_source_lines.append("    breakpoint()\n")
            new_source_lines.append("\n" * (starting_line_no - 2))

    new_source = "".join(new_source_lines)

    # Compile the new source and execute it in the function's global context
    code = compile(new_source, func.__code__.co_filename, "exec")
    func_globals = func.__globals__.copy()
    exec(code, func_globals)

    # Return the modified function
    return func_globals[func.__name__]
