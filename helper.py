import pandas as pd


def generate_create_table_statements(df):
    create_statements = []
    
    # Group by table_name
    grouped = df.groupby('table_name')
    
    # Iterate over each table
    for table_name, group in grouped:
        columns_definitions = []
        constraints_definitions = []
        
        # Iterate through each column in the group (table)
        for _, row in group.iterrows():
            # Construct column definition
            column_definition = f"{row['column_name']} {row['data_type']}"
            
            # Handle character max length for varchar
            if row['data_type'] == 'character varying' and pd.notna(row['character_maximum_length']):
                column_definition += f"({int(row['character_maximum_length'])})"
            
            # Handle nullable constraint
            if row['is_nullable'] == 'NO':
                column_definition += " NOT NULL"
            
            # Handle default values
            if pd.notna(row['column_default']):
                column_definition += f" DEFAULT {row['column_default']}"
            
            # Add the column definition to the list
            columns_definitions.append(column_definition)
            
            # Handle constraints (e.g., PRIMARY KEY, UNIQUE)
            if pd.notna(row['constraint_type']):
                if row['constraint_type'] == 'PRIMARY KEY':
                    constraints_definitions.append(f"PRIMARY KEY ({row['constraint_column']})")
                elif row['constraint_type'] == 'UNIQUE':
                    constraints_definitions.append(f"UNIQUE ({row['constraint_column']})")
        
        # Combine column definitions and constraints
        all_definitions = columns_definitions + constraints_definitions
        
        # Create the CREATE TABLE statement
        create_statement = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(all_definitions) + "\n);"
        
        # Add the statement to the list
        create_statements.append(create_statement)
    
    return create_statements


def get_schema(db):
    schema_query = """
        SELECT 
    c.table_name,
    c.column_name,
    c.data_type,
    c.is_nullable,
    c.column_default,
    c.character_maximum_length,
    tc.constraint_type,
    kcu.column_name AS constraint_column
FROM 
    information_schema.columns c
LEFT JOIN 
    information_schema.key_column_usage kcu 
    ON c.table_name = kcu.table_name 
    AND c.column_name = kcu.column_name
LEFT JOIN 
    information_schema.table_constraints tc
    ON kcu.constraint_name = tc.constraint_name
    AND c.table_name = tc.table_name
WHERE 
    c.table_schema = 'public'  -- Modify this if you want to target another schema
ORDER BY 
    c.table_name, c.ordinal_position;
    """
    
    df = pd.read_sql_query(schema_query,con=db._engine)
    create_table_queries = generate_create_table_statements(df)
    
    create_table_queries = "\n\n\n".join(create_table_queries)
    # print(create_table_queries)
    # Run the query using SQLDatabase's run method
    # return db.run(schema_query)
    return create_table_queries

