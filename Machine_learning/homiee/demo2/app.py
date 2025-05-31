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

class tool():
    self.median():
    
## computing:
## --> median price for {steet_name, locality_name} for requested year/quarter
## --> use recursive forecasting or "walk forward" prediction/imputation for NaN prices
## --> price prediced, errors!!!

## display:
## --> predicted price 
## --> last actual price
## --> error in $ and %
