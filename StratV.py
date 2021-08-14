from __future__ import print_function
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from httplib2 import Http
from apiclient.discovery import build
from oauth2client import file, client, tools
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import base64
import datetime
import io
from base64 import *
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory
from flask import Flask, render_template, request, redirect, url_for
import itertools
import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
from datetime import datetime
import os
import dash_table
from pprint import pprint
import csv
import math
from decimal import*
import numpy as np
from numpy import*
import random
#import matplotlib
from scipy.optimize import*
#import matplotlib.pyplot as plt
#from pandastable.core import Table
#from pandastable.data import TableModel
from datetime import datetime
#import os
#import sys
from googleapiclient import discovery
from scipy.optimize import *

#import pyqtgraph
#from PyQt5 import QtWidgets, QtCore
#from pyqtgraph import PlotWidget, plot
#import pyqtgraph as pg
# import sys  # We need sys so that we can pass argv to QApplication
# import os
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
#from matplotlib.figure import Figure

server = Flask(__name__)
app = dash.Dash(server=server)
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = None
creds = ServiceAccountCredentials.from_json_keyfile_name('transactionmanagerdash-3eedffe0ea0a.json', scope)
client = gspread.authorize(creds)
service = discovery.build('sheets', 'v4', credentials=creds)
spreadsheet_id = '18y6VUtIrwWYJhpPCIPQS_C7E4wvxKYd3tQ85GVfGqDE'
sheet = service.spreadsheets()
#Sheet_1 = client.open("Oil_Water_Relative_Permeability_data").sheet1
#Sheet_2 = client.open("Oil_Water_Relative_Permeability_data").sheet2
#data = pd.DataFrame(Sheet_1.get_all_records()).reset_index()
#SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
#DOCUMENT_ID = '1TN45bAT3dF5h2VPDp1En0Q4sye99o6B9tNbGseBqqTY'


def get_google_sheet(spreadsheet_id, range_name):
    """ Retrieve sheet data using OAuth credentials and Google Python API. """
    service = build('sheets', 'v4', http=creds.authorize(Http()))
    gsheet = sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    return gsheet


def gsheet2df(gsheet):
    """ Converts Google sheet data to a Pandas DataFrame.
    Note: This script assumes that your data contains a header file on the first row!

    Also note that the Google API returns 'none' from empty cells - in order for the code
    below to work, you'll need to make sure your sheet doesn't contain empty cells,
    or update the code to account for such instances.

    """
    header = gsheet.get('values', [])[0]
    values = gsheet.get('values', [])[1:]
    if not values:
        print('No data found.')
    else:
        all_data = []
        for col_id, col_name in enumerate(header):
            column_data = []
            for row in values:
                column_data.append(row[col_id])
            ds = pd.Series(data=column_data, name=col_name)
            all_data.append(ds)
        df = pd.concat(all_data, axis=1)
        return df

# Retrieving Relative permeability data(sheet1) from google sheet
RANGE_NAME_1 = 'RelativePermeabilityData'
gsheet_1 = get_google_sheet(spreadsheet_id, RANGE_NAME_1)
RPERM_data = gsheet2df(gsheet_1)
# print(RPERM_data)

# Retrieving reservoir bed data(sheet2) from google sheet
RANGE_NAME_2 = 'BedData'
gsheet_2 = get_google_sheet(spreadsheet_id, RANGE_NAME_2)
bed_data = gsheet2df(gsheet_2)
# print(bed_data)
# df["Date"] = pd.to_datetime(data["Date"])
# analysis_data = df.set_index('Date')
# last_date = df['Date'].iloc[-1].strftime("%B %d, %Y")

# now = datetime.now()
# date_time = now.strftime("%B %d, %Y")
# date_and_time = now.strftime("%B %d, %Y | %H:%M:%S")
external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]
# VALID_USERNAME_PASSWORD_PAIRS = {
#     'gas': 'pilotxlab'
# }

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                meta_tags=[
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                ],
                )
server = app.server

# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )

#bed_data = pd.read_csv('Permeability_Porosity_distribution_data.csv')
#RPERM_data = pd.read_csv('Oil_Water_Relative_Permeability_data.csv')
# ARRANGING THE DATA IN ORDER OF DECREASING PERMEABILITY.
bed_data_sort = bed_data.sort_values(by='PERMEABILITY', ascending=False)
PORO = np.array(bed_data_sort['POROSITY']).astype(float)
permeability_array = np.array(bed_data_sort['PERMEABILITY']).astype(float)
#print(permeability_array)
h = np.array(bed_data_sort['THICKNESS']).astype(float)
SW = np.array(RPERM_data['SW']).astype(float)
KRW = np.array(RPERM_data['KRW']).astype(float)
KRO = np.array(RPERM_data['KRO']).astype(float)

#KRW_1_SOR = np.interp(1-SOR, SW, KRW)
#KRO_SWI = np.interp(SWI, SW, KRO)
# EXTRACTING THE SORTED LAYER COLUMN
Layer_column = bed_data_sort['LAYER'].to_numpy()
Layer_table =  pd.DataFrame(Layer_column, columns = ['Layers'])
#==========================================================================================================================
#This code calculates the permeability ratio, ki/kn
List_of_permeability_ratio = []
for permeability_index in range(len(permeability_array)):
    List_of_permeability_ratio_subset = [][:-permeability_index]
    for index,permeability in enumerate(permeability_array):
        if permeability_index <= index:
            permaebility_ratio = permeability/permeability_array[permeability_index]
            List_of_permeability_ratio_subset.append(permaebility_ratio)
    List_of_permeability_ratio.append(List_of_permeability_ratio_subset)

List_of_permeability_ratio_DataTable = pd.DataFrame(List_of_permeability_ratio).transpose()

Average_porosity = '%.2f' % np.mean(PORO)

app.title = "StratV"
app.layout = html.Div(id="tm", children=[
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='DATA ENTRY', value='tab-1'),
        dcc.Tab(label='WATERFLOOD METHODS', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])


@app.callback(
    Output('tabs-content', 'children'),
    [
        Input('tabs', 'value'),
    ],
)
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(id='tabular-data', children=[
            html.Div(id='relative-permeability', children=[
                html.Iframe(src="https://docs.google.com/spreadsheets/d/18y6VUtIrwWYJhpPCIPQS_C7E4wvxKYd3tQ85GVfGqDE/edit?usp=sharing/pubhtml?widget=true&amp;headers=false"
                
                )
            ])
        ]
    ),
    elif tab == 'tab-2':
        return html.Div([
            html.Div(id='container1', children=[
                html.Aside(id="inputs",
                        children=[
                            html.H1('INPUTS'),
                            html.H5('(Field Units)'),
                            dcc.Input(id="number-of-points",
                                    placeholder='Number of Points',
                                    type='number',
                                    value=10
                                    ),
                            dcc.Input(id="length-of-bed",
                                    placeholder='Length of Bed',
                                    type='number',
                                    value=2896
                                    ),
                            dcc.Input(id="bed-width",
                                    placeholder='Width of Bed',
                                    type='number',
                                    value=2000
                                    ),
                            dcc.Input(id="porosity",
                                    placeholder='Average Porosity',
                                    type='number',
                                    value=0.25
                                    ),
                            dcc.Input(id="VISO",
                                    placeholder='Oil Viscosity',
                                    type='number',
                                    value=3.6
                                    ),
                            dcc.Input(id="VISW",
                                    placeholder='Water Viscosity',
                                    type='number',
                                    value=0.95
                                    ),
                            dcc.Input(id="OFVF",
                                    placeholder='Oil Formation Volume Factor',
                                    type='number',
                                    value=1.11
                                    ),
                            dcc.Input(id="WFVF",
                                    placeholder='Water Formation Volume Factor',
                                    type='number',
                                    value=1.01
                                    ),
                            dcc.Input(id="SWI",
                                    placeholder='Initial Water Saturation',
                                    type='number',
                                    value=0.2
                                    ),
                            dcc.Input(id="SGI",
                                    placeholder='Initial Gas Saturation',
                                    type='number',
                                    value=0.16
                                    ),
                            dcc.Input(id="SOI",
                                    placeholder='Initial Oil Saturation',
                                    type='number',
                                    value=0.64
                                    ),
                            dcc.Input(id="SOR",
                                    placeholder='Residual Oil Saturation',
                                    type='number',
                                    value=0.35
                                    ),
                            dcc.Input(id="CIR",
                                    placeholder='Constant Injection Rate',
                                    type='number',
                                    value=1800
                                    ),
                            dcc.Input(id="injection-pressure",
                                    placeholder='Injection Pressure',
                                    type='number',
                                    value=700
                                    ),
                            dcc.Input(id="RGSUZ",
                                    placeholder='Unswept Zone Residual Gas Saturation',
                                    type='number',
                                    value=0.06
                                    ),
                            dcc.Input(id="RGSSZ",
                                    placeholder='Swept Zone Residual Gas Saturation',
                                    type='number',
                                    value=0.02
                                    )
                            # html.Button('View Fractional Flow Plot', id='fractionalflow-button', n_clicks=0),
                            # html.Div(id='button-click', children='')

                        ]
                        )
                    ]),
                    html.Div(id='container2', children=[
                        html.Aside(id='method', children=[
                            html.H1('WATERFLOOD METHODS'),
                            dcc.RadioItems(id="method-options",
                                options=[
                                    {'label': 'Roberts', 'value': 'Ro'}
                                ],
                                value=''
                            ),
                            html.Div(id='method-output',
                                children=''    
                            ),
                            html.H5(children="Chart type"),
                            dcc.Dropdown(id="chart-type",className="inputs",
                                options=[
                                    {'label': 'line',
                                        'value': 'line'},
                                    {'label': 'bar',
                                        'value': 'bar'},
                                    {'label': 'scatter',
                                        'value': 'scatter'},
                                    {'label': 'pie',
                                        'value': 'pie'},
                                    {'label': 'sunburst',
                                        'value': 'sunburst'},
                                    {'label': 'sankey',
                                        'value': 'sankey'},
                                    {'label': 'pointcloud',
                                        'value': 'pointcloud'},
                                    {'label': 'treemap',
                                        'value': 'treemap'},
                                    {'label': 'table',
                                        'value': 'table'},
                                    {'label': 'scattergl',
                                        'value': 'scattergl'}
                                    
                                ],
                                value='',
                                placeholder="Chart type",
                                clearable=True
                            ),
                            html.Div(id='x-axis',
                                children=''    
                            ),
                        ]

                        )
                    ]),
                     html.H2('General Calculations'),
                     html.Div(id='general-output',
                         children=[
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Average Porosity'),
                                     html.Div(className='V',id='average-porosity',children=str(Average_porosity)+' cp')
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Relative Permeability at 1-SOR'),
                                     html.Div(className='V',id='relative-perm-at-1-SOR',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Relative Permeability at SWI'),
                                     html.Div(className='V',id='relative-perm-at-SWI',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Gross Rock Volume'),
                                     html.Div(className='V',id='gross-rock-volume',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Displacement Efficiency'),
                                     html.Div(className='V',id='displacement-efficiency',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Mobility Ratio'),
                                     html.Div(className='V',id='mobility-ratio',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Average Sweep Efficiency at Breakthrough'),
                                     html.Div(className='V',id='average-sweep-efficiency-at-breakthrough',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Area of Reservoir Bed'),
                                     html.Div(className='V',id='area-of-reservoir',children=0.00)
                                 ]
                             ),
                             html.Div(className='general-calculation',
                                 children=[
                                     html.Div(className='P',children='Area Sweep Efficiency'),
                                     html.Div(className='V',id='area-sweep-efficiency',children=0.00)
                                 ]
                             ),
                         ]
                     ),
                     html.Div(className='table',children=[
                        html.H2('Roberts Tabular Result'),
                        html.Div(id='Robert-table-display',
                            children=''
                        )
                    ]
                    ),
                    html.Div(id='graph',
                        children=[
                        dcc.Graph(className="chart",
                            id="Robert-Chart", config={"displayModeBar": True},
                            style={'border': 'solid rgb(19, 18, 18)'}
                        )
                        ]
                    )
            ]
        )
      



@app.callback(
    Output("relative-perm-at-1-SOR", "children"),
    [
        Input("SOR", "value")
    ]
)
def relative_perm_1_SOR(SOR):
    KRW_1_SOR = '%.3f' % np.interp(1 - float(SOR), SW, KRW)
    return '{}'.format(KRW_1_SOR)

@app.callback(
    Output("relative-perm-at-SWI", "children"),
    [
        Input("SWI", "value")
    ]
)
def relative_perm_SWI(SWI):
    KRO_SWI = '%.3f' % np.interp(SWI, SW, KRO)
    return '{}'.format(KRO_SWI)


@app.callback(
    Output("mobility-ratio", "children"),
    [
        Input("SOR", "value"),
        Input("SWI", "value"),
        Input("VISO", "value"),
        Input("VISW", "value"),
    ]
)
def mobility_ratio(SOR,SWI,VISO,VISW):
    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Mobility_Ratio = '%.3f' % (KRW_1_SOR * VISO / (KRO_SWI * VISW))
    return '{}'.format(Mobility_Ratio)

@app.callback(
    Output("average-sweep-efficiency-at-breakthrough", "children"),
    [
        Input("SOR", "value"),
        Input("SWI", "value"),
        Input("VISO", "value"),
        Input("VISW", "value"),
    ]
)
def areal_sweep_efficiency_at_breakthrough(SOR,SWI,VISO,VISW):
    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Mobility_Ratio = KRW_1_SOR*VISO/(KRO_SWI*VISW)
    Areal_sweep_efficiency_at_breakthrough =  '%.3f' % (0.54602036+(0.03170817/Mobility_Ratio)+(0.30222997/math.exp(Mobility_Ratio)-0.0050969*Mobility_Ratio))
    return '{}'.format(Areal_sweep_efficiency_at_breakthrough)

@app.callback(
    Output("area-of-reservoir", "children"),
    [
        Input("length-of-bed", "value"),
        Input("bed-width", "value"),
    ]
)
def area_acres(Length_of_bed_ft,width_of_bed_ft):
    Area_acres =  '%.3f' % (Length_of_bed_ft*width_of_bed_ft/43560)
    return '{} acres'.format(Area_acres)

@app.callback(
    Output("gross-rock-volume", "children"),
    [
        Input("length-of-bed", "value"),
        Input("bed-width", "value"),
    ]
)
def gross_rock_volume(Length_of_bed_ft,width_of_bed_ft):
    Area_acres = Length_of_bed_ft*width_of_bed_ft/43560
    Gross_rock_volume_acre_ft =  '%.3f' % (Area_acres*h.sum())
    return '{} acres-ft'.format(Gross_rock_volume_acre_ft)
@app.callback(
    Output("displacement-efficiency", "children"),
    [
        Input("SOR", "value"),
        Input("SWI", "value"),
        Input("SGI", "value"),
    ]
)
def displacement_efficiency(SOR,SWI,SGI):
    Displacement_efficiency =  '%.3f' % ((1-SWI-SGI-SOR)/(1-SWI-SGI))
    return '{}'.format(Displacement_efficiency)

@app.callback(
    Output("area-sweep-efficiency", "children"),
    [
        Input("SOR", "value"),
        Input("SWI", "value"),
        Input("SGI", "value"),
        Input("VISO", "value"),
        Input("VISW", "value"),
    ]
)
def areal_sweep_efficiency(SOR,SWI,SGI,VISO,VISW):
    KRW_1_SOR = np.interp(1-SOR, SW, KRW)
    KRO_SWI = np.interp(SWI, SW, KRO)
    Mobility_Ratio = KRW_1_SOR*VISO/(KRO_SWI*VISW)
    Areal_sweep_efficiency_at_breakthrough = 0.54602036+(0.03170817/Mobility_Ratio)+(0.30222997/math.exp(Mobility_Ratio)-0.0050969*Mobility_Ratio)
    Displacement_efficiency = (1-SWI-SGI-SOR)/(1-SWI-SGI)
    Areal_sweep_efficiency =  '%.3f' % (Areal_sweep_efficiency_at_breakthrough+0.2749*np.log((1/Displacement_efficiency)))
    return '{}'.format(Areal_sweep_efficiency)


@app.callback(
    [Output("method-output", "children"),Output("x-axis", "children")],
    [
        Input("method-options", "value")
    ]
)




def choose_method(method):
    suppress_callback_exceptions=True
    #suppress_callback_exceptions=True
    if method=='Ro':
        method_table_list = html.Div(className='checklist_container',children=[dcc.Checklist(id='Robert-checklist',
                options=[
                    {'label': 'Fractional Flow Table', 'value': "Fractional_flow_table"},
                    {'label': 'Capacity', 'value': "Capacity"},
                    {'label': 'Fraction of Total Capacity', 'value': "Fraction_of_total_Capacity"},
                    {'label': 'Injection Rate Per Layer', 'value': "Injection_Rate_Per_Layer"},
                    {'label': 'Cumulative Water Injection Per Layer', 'value': "Cummulative_Water_Injection_Per_Layer_list"},
                    {'label': 'Oil Production Before Breakthrough', 'value': "Oil_Production_Before_Breakthrough"},
                    {'label': 'Oil Production Per Layer After Breakkthrough', 'value': "Oil_Production_Per_Layer_After_Breakthrough_list"},
                    {'label': 'Water Production Per Layer After Breakkthrough', 'value': "Water_Production_Per_Layer_After_Breakthrough_list"},
                    {'label': 'Water Oil Ratio', 'value': "WOR_table"},
                    {'label': 'Recovery at Breakthrough per Layer', 'value': "Recovery_At_Breakthrough_Per_Layer_list"},
                    {'label': 'Cumulative Water Injection per Layer at Breakthrough', 'value': "Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list"},
                    {'label': 'Time to Breakthrough for Each Layer', 'value': "Time_To_Breakthrough_For_Each_Layer_list"},
                    {'label': 'Oil Recovery to Each Point', 'value': "Oil_Recovery_To_Each_Point_List"},
                    {'label': 'Time to Each Point', 'value': "Time_To_Each_Point_list"},
                ],
                value=['Fractional_flow_table']
            )])
        x_axis_options=html.Div(className='checklist_container',children=[html.H5(children="x-axis"),dcc.Dropdown(id='Robert-xaxis-dropdown',
            options=[
                {'label': 'Capacity', 'value': "Capacity"},
                {'label': 'Fraction of Total Capacity', 'value': "Fraction_of_total_Capacity"},
                {'label': 'Injection Rate Per Layer', 'value': "Injection_Rate_Per_Layer"},
                {'label': 'Cumulative Water Injection Per Layer', 'value': "Cummulative_Water_Injection_Per_Layer_list"},
                {'label': 'Oil Production Before Breakthrough', 'value': "Oil_Production_Before_Breakthrough"},
                {'label': 'Oil Production Per Layer After Breakkthrough', 'value': "Oil_Production_Per_Layer_After_Breakthrough_list"},
                {'label': 'Water Production Per Layer After Breakkthrough', 'value': "Water_Production_Per_Layer_After_Breakthrough_list"},
                {'label': 'Water Oil Ratio', 'value': "WOR_table"},
                {'label': 'Recovery at Breakthrough per Layer', 'value': "Recovery_At_Breakthrough_Per_Layer_list,"},
                {'label': 'Cumulative Water Injection per Layer at Breakthrough', 'value': "Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list"},
                {'label': 'Time to Breakthrough for Each Layer', 'value': "Time_To_Breakthrough_For_Each_Layer_list"}
            ],
            value='Time_To_Breakthrough_For_Each_Layer_list',
            placeholder="X-axis",
            clearable=True
        )])
    return method_table_list, x_axis_options


@app.callback(
    [Output("Robert-table-display", "children"),
    Output("Robert-Chart", "figure")
    ],
    [
        Input("SWI", "value"),
        Input("VISO", "value"),
        Input("VISW", "value"),
        Input("OFVF", "value"),
        Input("length-of-bed", "value"),
        Input("bed-width", "value"),
        Input("CIR", "value"),
        Input("Robert-checklist", "value"),
        Input("chart-type", "value"),
        Input("Robert-xaxis-dropdown", "value"),
    ]
)

def Robert(SWI, VISO, VISW, OFVF, Length_of_bed_ft, width_of_bed_ft,
            Constant_injection_rate, Robert_checklist, chart_type, Robert_xaxis_dropdown):
    global Fractional_flow_table
    global Capacity
    global Fraction_of_total_Capacity
    global Injection_Rate_Per_Layer
    global Cummulative_Water_Injection_Per_Layer_list
    global Oil_Production_Before_Breakthrough
    global Oil_Production_Per_Layer_After_Breakthrough_list
    global Water_Production_Per_Layer_After_Breakthrough_list
    global Recovery_At_Breakthrough_Per_Layer_list
    global Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list
    global Time_To_Breakthrough_For_Each_Layer_list
    global Oil_Recovery_To_Each_Point_List
    global Time_To_Each_Point_list

    SW_table = pd.DataFrame(SW, columns = ['SW'])
    # Using the correlation between relative permeability ratio and water saturation

    # Calculating the coefficient b
    b = (np.log((KRO/KRW)[2])-np.log((KRO/KRW)[3]))/(SW[3]-SW[2])
    #========================================================================

    # Calculating the coefficient a
    a = (KRO/KRW)[2]*math.exp(b*SW[2])
    #========================================================================
    # Calculating the fractional flow
    def fw(SW):
        fw = 1/(1+a*(VISW/VISO)*np.exp(-b*SW))
        return(fw)
    #========================================================================
    ''' To calculate a suitable slope for the tangent to the fractional flow curve
    Drawn from the initial water saturation'''

    ''' STEP1: Generate a list of uniformly distributed random numbers from a water saturation
    # greater than the initial water saturation to 1'''
    xList = []
    for i in range(0, 10000):
        x = random.uniform(SWI+0.1, 1)
        xList.append(x) 
    xs = np.array(xList)

    '''STEP2: Calculate different slopes of tangents or lines intersecting the fractional
    flow curve using the array generated in step 1 as the water saturation.'''
    m = 1/((xs-SWI)*(1+(VISW/VISO)*a*np.exp(-b*xs)))

    '''STEP3: Calculate the maximum slope from different slopes generated in step 2.
    The value of this slope will be the slope of the tangent to the fractional flow
    curve.'''
    tangent_slope=max(m)

    #==========================================================================
    # Calculate the breakthrough saturation.
    Saturation_at_Breakthrough = SWI + 1/tangent_slope
    
    #===========================================================================
    # Calculating the saturation at the flood front

    def funct(SWF):
        swf = SWF[0]
        F = np.empty((1))
        F[0] = ((tangent_slope*(swf-SWI)*(1+(VISW/VISO)*a*math.exp(-b*swf)))-1)
        return F
    SWF_Guess = np.array([SWI+0.1])
    SWF = fsolve(funct, SWF_Guess)[0]
    #============================================================================
    # Fractional flow at the flood front
    Fwf = fw(SWF)
    #=============================================================================
    # Fractional flow
    Fw = fw(SW)
    Fw_table = pd.DataFrame(Fw, columns = ['Fractional Flow (Fw)'])
    #=============================================================================
    # Calculating the differential of the fractional flow equation
    def dFw_dSw(Sw):
        dfw_dSw = (VISW/VISO)*a*b*np.exp(-Sw*b)/(1+(VISW/VISO)*a*np.exp(-Sw*b))**2
        return dfw_dSw
    dfw_dSw_table = pd.DataFrame(dFw_dSw(SW), columns = ['dFw/dSw'])
    #============================================================================
    # Generating the data for the tangent plot
    tangent = (SW-SWI)*tangent_slope
    tangent_table = pd.DataFrame(tangent, columns = ['Tangent'])
    #==============================================================================
    '''Draw several tangents to the fractional flow curve at Sw values greater than the
    breakthrough saturation. Determine Sw and dFw/dSw and corresponding to these values.
    Plot fw’ versus Sw and construct a smooth curve through the points '''
    # Sw greater than SwBT
    Sw_greater_SwBT = arange(Saturation_at_Breakthrough+0.01,SW[len(SW)-1],0.01)
    dFw_dSw_greater_SwBT = dFw_dSw(Sw_greater_SwBT)
    #============================================================================
    Fractional_flow_table = pd.concat([SW_table, Fw_table, dfw_dSw_table, tangent_table], axis=1)
    #=============================================================================
            
    
    # class MainWindow(QtWidgets.QMainWindow):
    
    #     def __init__(self, *args, **kwargs):
    #         super(MainWindow, self).__init__(*args, **kwargs)
    
    #         self.graphWidget = pg.PlotWidget()
    #         self.setCentralWidget(self.graphWidget)
    
    #         #Add Background colour to white
    #         self.graphWidget.setBackground('w')
    #         # Add Title
    #         self.graphWidget.setTitle("Fractional Flow Curve", color="b", size="20pt")
    #         # Add Axis Labels
    #         styles = {"color": "#f00", "font-size": "18px"}
    #         self.graphWidget.setLabel("left", "Fractional Flow (Fw)", **styles)
    #         self.graphWidget.setLabel("right", "Differential of Fractional Flow (dFw/dSw)", **styles)
    #         self.graphWidget.setLabel("bottom", "Water Saturation (Sw)", **styles)
    #         #Add legend
    #         self.graphWidget.addLegend()
    #         #Add grid
    #         self.graphWidget.showGrid(x=True, y=True)
    #         #Set Range
    #         self.graphWidget.setXRange(0, 1, padding=0)
    #         self.plot(SW, fw(SW), "Fw", 'r')
    #         self.plot(SW, tangent, "Tangent", 'k')
    #         self.plot(SW, dfw_dSw, "dFw/dSw", 'b')

    
    #     def plot(self, x, y, plotname, color):
    #         pen = pg.mkPen(color=color)
    #         self.graphWidget.plot(x, y, name=plotname, pen=pen, symbolBrush=(color))
    
    # def main():
    #     app = QtWidgets.QApplication(sys.argv)
    #     main = MainWindow()
    #     main.show()
    #     main._exit(app.exec_())
    #     #QApplication.exec_()
    # if __name__ == '__main__':
    #     main()
            

    #=============================================================================
    #Calculating capacity 
    permeability = bed_data['PERMEABILITY'].astype(float)
    thickness = bed_data['THICKNESS'].astype(float)
    porosity = bed_data['POROSITY'].astype(float)
    Capacity=permeability*thickness
    Fraction_of_total_Capacity= Capacity/sum(Capacity)
    #=============================================================================
    #Calculating Injection Rate per layer
    Injection_Rate_Per_Layer = Constant_injection_rate*Fraction_of_total_Capacity

    #=============================================================================
    #Calculating Water injection rate per layer
    Area = Length_of_bed_ft*width_of_bed_ft/43560

    Cummulative_Water_Injection_Per_Layer_list = []
    for j in range(len(thickness)):
        Cummulative_Water_Injection_Per_Layer = 7758*Area*thickness[j]*porosity[j]/dFw_dSw_greater_SwBT
        Cummulative_Water_Injection_Per_Layer_list.append(Cummulative_Water_Injection_Per_Layer)

    #=============================================================================
    #Oil Production Rate Before Breakthrough
    Oil_Production_Before_Breakthrough = Injection_Rate_Per_Layer/OFVF

    #Oil Production Rate After Breakthrough
    Oil_Production_Per_Layer_After_Breakthrough_list = []
    for j in range(len(thickness)):
        Oil_Production_Per_Layer_After_Breakthrough = Oil_Production_Before_Breakthrough[j]*(1-Fw)
        Oil_Production_Per_Layer_After_Breakthrough_list.append(Oil_Production_Per_Layer_After_Breakthrough)

    #Water Production 
    Water_Production_Per_Layer_After_Breakthrough_list = []
    for j in range(len(thickness)):
        Water_Production_Per_Layer_After_Breakthrough = Injection_Rate_Per_Layer[j]*Fw
        Water_Production_Per_Layer_After_Breakthrough_list.append(Water_Production_Per_Layer_After_Breakthrough)    

    #Calculate the recovery at breakthrough and the time to breakthrough for each layer
    Recovery_At_Breakthrough_Per_Layer_list = []
    for j in range(len(thickness)):    
        Recovery_At_Breakthrough_Per_Layer = 7758*Area*thickness[j]*porosity[j]*(Saturation_at_Breakthrough - SWI)/OFVF
        Recovery_At_Breakthrough_Per_Layer_list.append(Recovery_At_Breakthrough_Per_Layer)
    #print(Recovery_At_Breakthrough_Per_Layer_list)   
    #Time to Breakthrough for each layer
    #Water injection at Breakthrough
    Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list = []
    for j in range(len(thickness)):
        Cummulative_Water_Injection_Per_Layer_At_Breakthrough = 7758*Area*thickness[j]*porosity[j]/dFw_dSw(Saturation_at_Breakthrough)
        Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list.append(Cummulative_Water_Injection_Per_Layer_At_Breakthrough)   

    Time_To_Breakthrough_For_Each_Layer_list = []
    for j in range(len(thickness)):
        Time_To_Breakthrough_For_Each_Layer = Cummulative_Water_Injection_Per_Layer_At_Breakthrough/Injection_Rate_Per_Layer[j]
        Time_To_Breakthrough_For_Each_Layer_list.append(Time_To_Breakthrough_For_Each_Layer)    

    #Oil recovery and time to each point.
    Oil_Recovery_To_Each_Point_List = []
    for j in range(len(thickness)):
        Oil_Recovery_To_Each_Point =  7758*Area*thickness[j]*porosity[j]*(SW - SWI)/OFVF
        Oil_Recovery_To_Each_Point_List.append(Oil_Recovery_To_Each_Point)

    Time_To_Each_Point_list = []
    for j in range(len(thickness)):    
        Time_To_Each_Point = Cummulative_Water_Injection_Per_Layer/Injection_Rate_Per_Layer[j]
        Time_To_Each_Point_list.append(Time_To_Each_Point)
    
    Capacity = pd.DataFrame(Fraction_of_total_Capacity).rename(columns={0: 'Capacity'})
    Fraction_of_total_Capacity = pd.DataFrame(Fraction_of_total_Capacity).rename(columns={0: 'Fraction of Total Capacity'})
    Injection_Rate_Per_Layer=pd.DataFrame(Injection_Rate_Per_Layer).rename(columns = {0:'Injection Rate per Layer'})
    Oil_Production_Before_Breakthrough = pd.DataFrame(Injection_Rate_Per_Layer).rename(columns={0: 'Oil Production Before Breakthrough'})
    Recovery_At_Breakthrough_Per_Layer_list = pd.DataFrame(Recovery_At_Breakthrough_Per_Layer_list).rename(columns={0: 'Recovery At Breakthrough Per Layer'})
    Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list=pd.DataFrame(Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list).rename(columns = {0:'Cumulative Water Injection Per Layer At Breakthrough'})
    Time_To_Breakthrough_For_Each_Layer_list = pd.DataFrame(Time_To_Breakthrough_For_Each_Layer_list).rename(columns={0: 'Time To Breakthrough For Each Layer'})
    #WOR_table = pd.DataFrame(WOR_list).rename(columns={0: 'Water Oil Ratio'})


    Robert_data_list = [Fractional_flow_table, Capacity, Fraction_of_total_Capacity, Injection_Rate_Per_Layer,
    Cummulative_Water_Injection_Per_Layer_list, Oil_Production_Before_Breakthrough, Oil_Production_Per_Layer_After_Breakthrough_list,
    Water_Production_Per_Layer_After_Breakthrough_list, Recovery_At_Breakthrough_Per_Layer_list,
    Cummulative_Water_Injection_Per_Layer_At_Breakthrough_list, Time_To_Breakthrough_For_Each_Layer_list,
    Oil_Recovery_To_Each_Point_List, Time_To_Each_Point_list]
    
    translation = {39: None}
    Robert_checklists = str(Robert_checklist).translate(translation)
    def variablename(var):
        return [tpl[0] for tpl in filter(lambda x:var is x[1], globals().items())]
    
    #Collecting DataFrame for Table
    l_b=pd.DataFrame()
    for j in Robert_checklist:
        for d in Robert_data_list:
            if variablename(d)[0] == j:
                l_b = l_b.append(d,ignore_index=True)     
    translation = {39: None}
    Robert_xaxis_dropdowns = str(Robert_xaxis_dropdown).translate(translation)

    #Collecting Data for graph x-axis
    l_bxaxis = pd.DataFrame()
    for q in Robert_data_list:
        if variablename(q)[0] == Robert_xaxis_dropdowns:
            l_bxaxis=l_bxaxis.append(q,ignore_index=True)

    #Collecting Data for graph y-axis
    lb = []
    for j in l_b.values.tolist():
        for m in j:
            lb.append(m)
    lbxaxis = []
    for j in l_bxaxis.values.tolist():
        for m in j:
            lbxaxis.append(m)

    #The code integrates marker status
    if chart_type == 'line':
        marker = None
    else:
        marker='markers'
    Robert_Chart = {
        "data": [
            {
                "x": lbxaxis,
                "y": lb,
                "type": str(chart_type),
                "mode": marker,
            },
        ],
        "layout": {
            "title": {"text": 'Robert: ' + str(Robert_checklists) +' vs ' + str(Robert_xaxis_dropdowns), "x": 0.05, "xanchor": "left"},
            "xaxis": {"title":str(Robert_xaxis_dropdowns),"fixedrange": True},
            "yaxis": {"title":str(Robert_checklists),"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }
    return dash_table.DataTable(
                            columns=[{"name": str(i), "id": str(i)} for i in l_b.columns],
                            data=l_b.to_dict('records'),
                            editable=True,
                            style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                            style_cell={
                                'backgroundColor': '#54c5f9',
                                'color': 'white'
                            }
                            ),Robert_Chart
                                


if __name__ == "__main__":
    app.run_server(debug=False)
