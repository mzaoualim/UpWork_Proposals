'''
Stage 1: A tool that can estimate what a house / apartment should sell at each year 
with basic understanding of inflation and which suburbs / streets are more expensive than others. 
(current engagement)

Tasks:

    Exclude all commercial data, rural data etc.
    Understand house data vs apartment data
    Be aware of duplex data
    Run model based on last 10 years for recent growth
    Use three suburbs as a test case e.g. Panania, Chatswood / Chatswood West


'''
def data_model_loader():
    # choose df_all or df_10y --> automalically switch for appropriate model
    data, model = (), ()
    return data, model

def chooser():
    # choices
    ## --> input {unit_number, house_number} + {year, quarter}
    ## --> choices {street_name, locality_name} ==> from saved json dictionnary
    ## --> fixed {state}
    choise_dict = {}
    return choise_dict

def median_computer(data, street, locality, year, quarter):
    median = data.groupby([street, locality, year, quarter])['Price'].median()
    return median


def recursive_forcaster():
    model, _ = data_model_loader()
    inputs = chooser()
    median = median_computer()
    predicted_price = model.predict(data)

    return predicted_price

## display:
## --> predicted price 
## --> last actual price
## --> error in $ and %
