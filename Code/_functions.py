
def clean_data(df):
    # @author TP
    """This function is used to remove irrelevant data before input to model"""
    relevant_data = df['comment'].dropna()
    return relevant_data[1:]
    
