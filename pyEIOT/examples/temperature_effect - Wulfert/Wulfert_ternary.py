from plotly.offline import plot
import plotly.graph_objs as go
import scipy.io as spio
Wulfert_Data      = spio.loadmat('flodat.mat')

conc=Wulfert_Data['conc']


        
rawData = [
    {'Ethanol':1,        'Water':0,         'Isopropanol':0,         'label':'point 1'},
    {'Ethanol':0.664441, 'Water':0.335559,  'Isopropanol':0,         'label':'point 2'},
    {'Ethanol':0.671541, 'Water':0.163104,  'Isopropanol':0.165355,  'label':'point 3'},
    {'Ethanol':0.666259, 'Water':0,         'Isopropanol':0.333741,  'label':'point 4'},
    {'Ethanol':0.499822, 'Water':0.500178,  'Isopropanol':0,         'label':'point 5'},
    {'Ethanol':0.500252, 'Water':0.333047,  'Isopropanol':0.166701,  'label':'point 6'},
    {'Ethanol':0.499427, 'Water':0.167176,  'Isopropanol':0.333397,  'label':'point 7'},
    {'Ethanol':0.500262, 'Water':0,         'Isopropanol':0.499738,  'label':'point 8'},
    {'Ethanol':0.333231, 'Water':0.666769,  'Isopropanol':0,         'label':'point 9'},
    {'Ethanol':0.33245,  'Water':0.500332,  'Isopropanol':0.167218,  'label':'point 10'},
    {'Ethanol':0.332835, 'Water':0.334015,  'Isopropanol':0.333149,  'label':'point 11'},
    {'Ethanol':0.322196, 'Water':0.16555,   'Isopropanol':0.512254,  'label':'point 12'},
    {'Ethanol':0.335115, 'Water':0,         'Isopropanol':0.664885,  'label':'point 13'},
    {'Ethanol':0.166319, 'Water':0.666851,  'Isopropanol':0.166829,  'label':'point 14'},
    {'Ethanol':0.167002, 'Water':0.500005,  'Isopropanol':0.332993,  'label':'point 15'},
    {'Ethanol':0.166248, 'Water':0.333134,  'Isopropanol':0.500618,  'label':'point 16'},
    {'Ethanol':0.162242, 'Water':0.163008,  'Isopropanol':0.67475,   'label':'point 17'},
    {'Ethanol':0,        'Water':1,         'Isopropanol':0,         'label':'point 18'},
    {'Ethanol':0,        'Water':0.667071,  'Isopropanol':0.332929,  'label':'point 19'},
    {'Ethanol':0,        'Water':0.499685,  'Isopropanol':0.500315,  'label':'point 20'},
    {'Ethanol':0,        'Water':0.333868,  'Isopropanol':0.666132,  'label':'point 21'},
    {'Ethanol':0,        'Water':0,         'Isopropanol':1,         'label':'point 22'},
];



def makeAxis(title, tickangle):
    return {
      'title': title,
      'titlefont': { 'size': 20 },
      'tickangle': tickangle,
      'tickfont': { 'size': 15 },
      'tickcolor': 'rgba(0,0,0,0)',
      'ticklen': 5,
      'showline': True,
      'showgrid': True
    }

fig = go.Figure(go.Scatterternary({
    'mode': 'markers',
    'a': [i for i in map(lambda x: x['Ethanol'], rawData)],
    'b': [i for i in map(lambda x: x['Water'], rawData)],
    'c': [i for i in map(lambda x: x['Isopropanol'], rawData)],
    'text': [i for i in map(lambda x: x['label'], rawData)],
    'marker': {
        'symbol': 100,
        'color': '#DB7365',
        'size': 14,
        'line': { 'width': 2 }
    }
}))

fig.update_layout({
    'ternary': {
        'sum': 100,
        'aaxis': makeAxis('Ethanol', 0),
        'baxis': makeAxis('<br>Water', 45),
        'caxis': makeAxis('<br>Isopropanol', -45)
    },
    'annotations': [{
      'showarrow': False,
      'text': 'Simple Ternary Plot with Markers',
        'x': 0.5,
        'y': 1.3,
        'font': { 'size': 15 }
    }]
})

plot(fig)