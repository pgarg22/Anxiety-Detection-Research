#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 22:30:09 2022

@author: octopusphoenix
"""




from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import numpy as np
import plotly.express as px

colors_scale_options=["greens","oranges"]


colors_scale_options= [[(0.00, "#e6f69d"),   (0.11, "#e6f69d"),
                         (0.11, "#aadea7"), (0.24, "#aadea7"),
                         (0.24, "#64c2a6"),  (0.4, "#64c2a6"),
                         (0.4, "#2d87bb"),  (1.00, "#2d87bb")],
                        [(0.00, "#fef001"),   (0.125, "#fef001"),
                         (0.125, "#fd9a01"), (0.25, "#fd9a01"),
                         (0.25, "#fd6104"), (0.41, "#fd6104"),
                         (0.41, "#F00505"),  (1.00, "#ff2c05")], ] 

def push_viz_scatter_subplots(Number, df,title, subtitles,ytitle,mode,symbol,range_max,tick_text):
    
    
    fig = make_subplots(rows=1, cols=Number,horizontal_spacing = 0,
                    vertical_spacing= 0.20,
                    subplot_titles=subtitles, y_title=ytitle,shared_yaxes=True)
    
    cs_place=1.07
    for i in range(1,Number+1):
        fig.append_trace(go.Scatter(
        x=df.iloc[:, 0],
        y=df.iloc[:, i],
        name=df.columns[i],
        mode=mode,
        marker=dict(
            symbol=symbol[i-1], line=dict(width=2, color="DarkSlateGrey"),
           size=df.iloc[:, i]+15,
           color=df.iloc[:, i], #set color equal to a variable
           colorscale=colors_scale_options[i-1],
           colorbar=dict( x= cs_place, title=df.columns[i],
           tickvals=[4,12,20,40],
           ticktext=tick_text[i-1],
           lenmode="pixels", len=300, ),
            # one of plotly colorscales
           showscale=True,
           cmax=range_max[i-1],
           cmin=0)
        
        ), row=1, col=i)
        cs_place=cs_place+0.2
        
    
    fig.update_traces( marker_line_width=2)
    fig.update_layout(height=800, width=1600, title=title,
                      xaxis_title= df.columns[0],
                
                      yaxis_zeroline=False, xaxis_zeroline=False, 
                      )
    fig.show()

    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/"+title+".png")
    


    

    
    
    
    
def push_viz_bar_subplots(Number, df,title, subtitles,ytitle,range_max,tick_text):
    
    cs_place=1.07
    fig = make_subplots(rows=1, cols=Number,horizontal_spacing = 0.15,
                    vertical_spacing= 0.20,
                    subplot_titles=subtitles)
    for i in range(1,Number+1):
        df= df.sort_values(by=df.columns[i])
        fig.append_trace(go.Bar(
        x=df.iloc[:, 0],
        y=df.iloc[:, i],
        name=df.columns[i],
        marker=dict(
           
           color=df.iloc[:, i], #set color equal to a variable
           colorscale=colors_scale_options[i-1],
           colorbar=dict( x= cs_place, title=df.columns[i],
           tickvals=[4,12,20,40],
           ticktext=tick_text[i-1],
           lenmode="pixels", len=300, ),
            # one of plotly colorscales
           showscale=True,
           cmax=range_max[i-1],
           cmin=0)), row=1, col=i)

        cs_place=cs_place+0.2
    

    fig.update_traces( marker_line_width=2)
    fig.update_layout(height=800, width=1600, title=title,
                  xaxis_title= df.columns[0],
                  xaxis={'categoryorder':'array', 'categoryarray':['<10', '10-20', '20-30','30-40', '40-50','50-60','60+']},
                  yaxis_zeroline=False, xaxis_zeroline=False, 
                  )
    fig.show()
    
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/"+title+".png")
    

def push_viz_scatter(Number, df,title,ytitle,mode,symbol,range_max,tick_text):
    
    
    fig = go.Figure()
    cs_place=1.07
    
    for i in range(1,Number+1):
        
        
    
        fig.add_trace(go.Scatter(
        x=df.iloc[:, 0],
        y=df.iloc[:, i],
        name=df.columns[i],
        mode=mode,
        marker=dict(
            symbol=symbol[i-1], line=dict(width=2, color="DarkSlateGrey"),
           size=df.iloc[:, i]+15,
           color=df.iloc[:, i], #set color equal to a variable
           colorscale=colors_scale_options[i-1],
           colorbar=dict( x= cs_place, title=df.columns[i],
           tickvals=[4,12,20,40],
           ticktext=tick_text[i-1],
           lenmode="pixels", len=300, ),
            # one of plotly colorscales
           showscale=True,
           cmax=range_max[i-1],
           cmin=0)))
        cs_place=cs_place+0.2

        
    fig.update_traces( marker_line_width=2)
    fig.update_layout(height=800, width=1600, title=title,
                      xaxis_title= df.columns[0],
                      yaxis_title=ytitle,
                      legend_title="Legend",
                      yaxis_zeroline=False, xaxis_zeroline=False)
    fig.show()
    
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/"+title+".png")



def push_viz_bar(Number, df,title, subtitles,ytitle,range_max,tick_text):
    
    
    fig = go.Figure()
    cs_place=1.07
    
    for i in range(1,Number+1):
        
        df= df.sort_values(by=df.columns[i])
    
        fig.add_trace(go.Bar(
        x=df.iloc[:, 0],
        y=df.iloc[:, i],
        name=df.columns[i],
        marker=dict( 
           color=df.iloc[:, i], #set color equal to a variable
           colorscale=colors_scale_options[i-1],
           colorbar=dict( x= cs_place, title=df.columns[i],
           tickvals=[4,12,20,40],
           ticktext=tick_text[i-1],
           lenmode="pixels", len=300, ),
            # one of plotly colorscales
           showscale=True,
           cmax=range_max[i-1],
           cmin=0)))
        cs_place=cs_place+0.2

        
    fig.update_traces( marker_line_width=2)
    fig.update_layout(height=800, width=1600, title=title,
                      xaxis_title= df.columns[0],
                      yaxis_title=ytitle,
                      legend_title="Legend",
                  yaxis_zeroline=False, xaxis_zeroline=False,xaxis={ 'categoryorder':'array', 'categoryarray':['<10', '10-20', '20-30','30-40', '40-50','50-60','60+']})
    fig.show()
    
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/"+title+".png")
    
  

def push_heatmap(z,modname):
    fig = px.imshow(z, text_auto=True, aspect="auto")
    fig.show()
    
    if not os.path.exists("heatmaps"):
        os.mkdir("heatmaps")
    fig.write_image("heatmaps/"+modname+".png")

    





def push_viz_scatter2(Number, df,title,ytitle,mode,symbol):
    
    
    fig = go.Figure()
    cs_place=1.07
    
    for i in range(1,Number+1):
        
        
    
        fig.add_trace(go.Scatter(
        x=df.iloc[:, 0],
        y=df.iloc[:, i],
        name=df.columns[i],
        mode=mode,
        marker=dict(
            symbol=symbol[i-1], line=dict(width=2, color="DarkSlateGrey"),
           color=df.iloc[:, i], #set color equal to a variable
            )))
            # one of plotly colorscales
       

        
    fig.update_traces( marker_line_width=2)
    fig.update_layout(height=800, width=1600, title=title,
                      xaxis_title= df.columns[0],
                      yaxis_title=ytitle,
                      legend_title="Legend",
                      yaxis_zeroline=False, xaxis_zeroline=False)
    fig.show()
    
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/"+title+".png")


def push_viz_scatter_subplots2(row,col, df,title,subtitles,ytitle,mode,symbol):
    
    
    fig = make_subplots(rows=row, cols=col,horizontal_spacing = 0,
                    vertical_spacing= 0.20,
                    subplot_titles=subtitles, y_title=ytitle,shared_yaxes=True)
    
    for i in range(1,(row)+1):
        for j in range(1,(col)+1):
            fig.append_trace(go.Scatter(
            x=df.iloc[:, 0],
            y=df.iloc[:, i],
            name=df.columns[i],
            mode=mode,
            marker=dict(
                symbol=symbol[i-1], line=dict(width=2, color="DarkSlateGrey"),
               color=df.iloc[:, i], #set color equal to a variable
          )
              
            
            ), row=i, col=j)
        
    
    fig.update_traces( marker_line_width=2)
    fig.update_layout(height=800, width=1600, title=title,
                      xaxis_title= df.columns[0],
                
                      yaxis_zeroline=False, xaxis_zeroline=False, 
                      )
    fig.show()

    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image("images/"+title+".png")
    


    

    
