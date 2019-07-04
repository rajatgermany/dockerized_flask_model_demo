import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import sys

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly import tools

from sklearn.metrics import mean_squared_error
from math import sqrt


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
colors = {"background": "white", "text": "#7FDBFF"}

### Train Data
train = pd.read_csv("meter-data_baseline.csv")
train.drop("Unnamed: 0", axis=1, inplace=True)
train.drop("dayofweek", axis=1, inplace=True)
train["deviceId"] = train.deviceId.astype("str")

# #### Results data
data_1 = pd.read_csv("week_23_results_2.csv")
data_2 = pd.read_csv("week_23_results.csv")
data_3 = pd.read_csv("week_23_results_3.csv")

model_results = pd.concat([data_1, data_2, data_3], ignore_index=True)
model_results["deviceId"] = model_results.deviceId.astype("str")


#### Error Analysis
intrested_hours = [20, 9, 13, 6, 3, 14, 2]

### Types of Plots
TYPE_OF_PLOTS = ["Profile", "Prediction", "Error"]

DAY_OF_WEEK = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
}

### App
app = dash.Dash()
app.config["suppress_callback_exceptions"] = True

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="Energy Model",
            style={
                "textAlign": "center",
                "color": colors["text"],
                "background": "green",
            },
        ),
        html.H2(children="Profiling", style={"textAlign": "center", "color": "red"}),
        html.Div(
            id="Profiling",
            style={"textAlign": "center", "color": colors["text"]},
            children=[
                dcc.Graph(id="graph_profile"),
                dcc.Dropdown(
                    options=[
                        {"label": deviceId, "value": deviceId}
                        for deviceId in list(train.deviceId.unique())
                    ],
                    id="profile_deviceId",
                    style={
                        "textAlign": "center",
                        "margin": "0 auto",
                        "width": "500px",
                        "height": "50px",
                    },
                    placeholder="DeviceId",
                    value=list(train.deviceId.unique())[0],
                ),
            ],
        ),
        #### Prediction
        html.H2(
            children="Prediction",
            style={"textAlign": "center", "color": "red", "margin-top": "200px"},
        ),
        html.Div(
            id="Prediction",
            style={"textAlign": "center", "color": colors["text"]},
            children=[
                dcc.Graph(id="graph_Prediction"),
                dcc.Dropdown(
                    options=[
                        {"label": deviceId, "value": deviceId}
                        for deviceId in list(model_results.deviceId.unique())
                    ],
                    id="deviceId_prediction",
                    style={
                        "textAlign": "center",
                        "margin": "0 auto",
                        "width": "500px",
                        "height": "50px",
                    },
                    placeholder="DeviceId",
                    value=list(model_results.deviceId.unique())[0],
                ),
                dcc.Dropdown(
                    options=[
                        {"label": label, "value": value}
                        for label, value in DAY_OF_WEEK.items()
                    ],
                    id="day_prediction",
                    style={
                        "textAlign": "center",
                        "margin": "0 auto",
                        "width": "500px",
                        "height": "50px",
                    },
                    placeholder="Enter Day",
                    value=6,
                ),
            ],
        ),
        html.H2(
            children="Error Analysis",
            style={"textAlign": "center", "color": "red", "margin-top": "200px"},
        ),
        ### Errro Analysis
        html.Div(
            id="Error-Analysis",
            style={"textAlign": "center", "color": colors["text"]},
            children=[
                dcc.Graph(id="graph_error_analysis"),
                dcc.Dropdown(
                    options=[
                        {"label": hour, "value": hour} for hour in intrested_hours
                    ],
                    value=intrested_hours[0],
                    id="hour_error_analysis",
                    style={
                        "textAlign": "center",
                        "margin": "0 auto",
                        "width": "500px",
                        "height": "50px",
                    },
                    placeholder="Hour",
                ),
                dcc.Dropdown(
                    value="deviceId",
                    id="deviceId_error_analysis",
                    style={
                        "textAlign": "center",
                        "margin": "0 auto",
                        "width": "500px",
                        "height": "50px",
                    },
                ),
            ],
        ),
    ],
)


################ Profile Graphs
@app.callback(Output("graph_profile", "figure"), [Input("profile_deviceId", "value")])
def draw_profile_graph(deviceId):
    device_profile_data = train[(train.deviceId == deviceId)]
    fig = tools.make_subplots(
        rows=2, cols=2, subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4")
    )

    trace0 = go.Box(
        y=device_profile_data.value.values,
        x=device_profile_data.week.values,
        marker=dict(color="green"),
    )

    trace1 = go.Box(
        y=device_profile_data.value.values,
        x=device_profile_data.day.values,
        marker=dict(color="red"),
    )

    trace2 = go.Box(
        y=device_profile_data.value.values,
        x=device_profile_data.day_of_week.values,
        marker=dict(color="black"),
    )

    trace3 = go.Box(
        y=device_profile_data.value.values,
        x=device_profile_data.hour.values,
        marker=dict(color="#3D9970"),
    )

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace3, 2, 2)

    fig["layout"]["xaxis1"].update(title="week")
    fig["layout"]["xaxis2"].update(title="day")
    fig["layout"]["xaxis3"].update(title="day_of_week")
    fig["layout"]["xaxis4"].update(title="hour")
    fig["layout"].update(height=1500, width=2000)

    return fig


################ Prediction Graphs


@app.callback(
    Output("graph_Prediction", "figure"),
    [Input("day_prediction", "value"), Input("deviceId_prediction", "value")],
)
def draw_prediction_graph(day_of_week, deviceId):
    print("day", day_of_week, deviceId, model_results.day_of_week)
    device_result_data = model_results[
        (model_results.deviceId == deviceId)
        & (model_results.day_of_week == day_of_week)
    ]

    trace1 = go.Scatter(
        y=device_result_data.value,
        x=device_result_data.index,
        mode="lines+markers",
        name="Actual Values",
    )
    trace2 = go.Scatter(
        y=device_result_data.Predicted,
        x=device_result_data.index,
        mode="lines+markers",
        name="Predicted Values",
    )
    new_data = [trace1, trace2]
    layout = go.Layout(
        yaxis=dict(title="Value"),
        boxmode="group",
        title="RMSE {}".format(
            sqrt(
                mean_squared_error(
                    device_result_data.value.values, device_result_data.Predicted.values
                )
            )
        ),
    )
    fig = go.Figure(data=new_data, layout=layout)
    return fig


################ Error-analysis Graphs


@app.callback(
    Output("deviceId_error_analysis", "options"),
    [Input("hour_error_analysis", "value")],
)
def set_hour(selected_hour):
    print("selected_hour", selected_hour)
    deviceIds = list(
        model_results[
            ((model_results.difference < -10) | (model_results.difference > 10))
            & (model_results.hour == selected_hour)
        ].deviceId.values
    )
    return [{"label": str(i), "value": str(i)} for i in deviceIds]


@app.callback(
    Output("graph_error_analysis", "figure"),
    [Input("hour_error_analysis", "value"), Input("deviceId_error_analysis", "value")],
)
def draw_error_analysis(hour_error_analysis, deviceId_error_analysis):
    train["deviceId"] = train.deviceId.astype("str")

    actual_data = train[
        (train.deviceId == deviceId_error_analysis)
        & (train.hour == hour_error_analysis)
    ].reset_index()
    predicted_data = model_results[
        (model_results.deviceId == deviceId_error_analysis)
        & (model_results.hour == hour_error_analysis)
    ].reset_index()

    print(predicted_data)

    trace1 = go.Scatter(
        y=actual_data.value,
        x=actual_data.index,
        mode="lines+markers",
        name="Actual Values",
    )
    trace2 = go.Scatter(
        y=predicted_data.Predicted,
        x=predicted_data.index,
        mode="lines+markers",
        name="Predicted Values",
    )

    new_data = [trace1, trace2]
    layout = go.Layout(
        yaxis=dict(title="Value"),
        boxmode="group",
        title="Predicted {},  Max {},  Mean {}, Min {}".format(
            predicted_data.Predicted.values[0],
            actual_data.value.max(),
            actual_data.value.mean(),
            actual_data.value.min(),
        ),
    )
    fig = go.Figure(data=new_data, layout=layout)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True, port=9990)
