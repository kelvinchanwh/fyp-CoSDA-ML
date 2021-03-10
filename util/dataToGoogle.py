# authenticate
from google.colab import auth
import gspread
from oauth2client.client import GoogleCredentials as GC
# create, and save df
from gspread_dataframe import set_with_dataframe, get_as_dataframe
import numpy as np
import pandas as pd

auth.authenticate_user()
gc = gspread.authorize(GC.get_application_default())
title = 'CoSDA-ML'
gc.create(title)  # if not exist
sheet = gc.open(title).sheet1

def getSheet(sheet):
    df = get_as_dataframe(sheet).dropna(0, 'all').dropna(1, 'all')
    return df

def writeSheet(df):
    set_with_dataframe(sheet, df)

def initSheet(x_div, y_div):
    df = pd.DataFrame(np.ones((x_div, y_div)), columns=[a*0.1 for a in range(x_div)])
    writeSheet(df)
    return sheet

def updateSheet(x, y, val):
    df = getSheet(sheet)
    df[x][y] = val
    writeSheet(df)

