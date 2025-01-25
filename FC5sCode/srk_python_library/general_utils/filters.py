def apply_filters(dataframe, filters):
    """
    Applies the filters on the dataframe according to the conditions specified in the 'filters' list.
    
    Parameters:
        
        dataframe (DataFrame): Pandas Dataframe
        filters (dict,tuple,list): Filters
                        If dict, it works just as the default Pandas filters
                        If tuple, it is possible to do exactly the same, but as tuple
                        If list, a list of tuples is expected. It is possible to include or exclude values.
                        Available operators are: ==, !=, <, <=, >, >=, contains or not_contains.
                        
    Example:
    
        data = {
                'Name': ['Ana', 'Juan', 'Luis', 'María', 'Marta'],
                'Age': [25, 30, 40, 22, 35],
                'Country': ['Argentina', 'México', 'Argentina', 'España', 'Colombia']
            }

        df = pd.DataFrame(data)
        filters = {'Country':'Argentina'}
        filters = {'Country':'Argentina','Age':25}
        filters = ('Country','Argentina')
        filters = [('Country','Argentina'),('Age',25)]
        filters = ('Country',('contains','o'))
        filters = [('Age', ('>=', 35, '|')), ('Age', ('<', 25))]
        filters = [('Age', ('>=', 25, '&')), ('Name', ('!=', ['Luis', 'Ana']))]
        filters = [('Age', ('>=', 25, '&')), ('Country', ('==', 'Argentina','&')),('Name',('!=','Ana'))]
        filters = [('Age', ('>=', 25, '&')), ('Name', ('not_contains', 'Lu'))]
        
    Returns:
    
        filtered_df (DataFrame): Filtered Pandas DataFrame
        
        """

    filter_str = ""
    if isinstance(filters, dict):
        if not filters:
            return dataframe
        else:
            for column, condition in filters.items():
                if isinstance(condition, str):
                    filter_str += f"{column} == '{condition}' & "
                else:
                    filter_str += f"{column} == {condition} & "
            filter_str = filter_str.rstrip(" & ")
            filtered_df = dataframe.query(filter_str)
            return filtered_df
    elif isinstance(filters, tuple) or isinstance(filters, list):
        if isinstance(filters, tuple):
            filters = [filters]
        for column, condition in filters:
            if isinstance(condition, str):
                filter_str += f"{column} == '{condition}' & "
            elif isinstance(condition, float) or isinstance(condition, int):
                filter_str += f"{column} == {condition} & "
            elif isinstance(condition, tuple) and (len(condition) == 2 or len(condition) == 3):
                symb = condition[0]
                filt = condition[1]
                if len(condition)==3:
                    and_or = condition[2]
                else:
                    and_or = '&'
                if symb in ['>','<','>=','<=','==','!=']:
                    if isinstance(filt,str):
                        filter_str += f"{column} {symb} '{filt}' {and_or} "
                    else:
                        filter_str += f"{column} {symb} {filt} {and_or} "
                elif symb == 'contains':
                    filter_str += f"{column}.str.contains('{filt}') {and_or} "
                elif symb == 'not_contains':
                    filter_str += f"~{column}.str.contains('{filt}') {and_or} "
                else:
                    print('Symbol not allowed')
                    return
            else:
                print('Please, check the filters')
        
        filter_str = filter_str.rstrip(" &|")
        filtered_df = dataframe.query(filter_str)
        filtered_df.reset_index(drop=True, inplace=True)
        return filtered_df