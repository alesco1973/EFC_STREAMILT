import streamlit as st
import pandas as pd
from statsbombpy import sb
import numpy as np
from mplsoccer import Sbopen, Pitch
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import Voronoi, voronoi_plot_2d




@st.cache_data(ttl=3600)
def merge_csv(csv_files):
    """
    Questa funzione unisce DataFrames Pandas basati sulla colonna comune "event_id".

    Args:
        csv_files (list): Un elenco di stringhe che rappresentano i percorsi dei file CSV.

    Returns:
        DataFrame: Il DataFrame risultante dall'unione dei DataFrame di input.
        
    Raises:
        ValueError: Se uno dei DataFrame di input non contiene la colonna "event_id".
    """
    
    if not csv_files:
        raise ValueError("Nessun file CSV fornito.")

    # Legge il primo file CSV come DataFrame di base
    base_df = pd.read_csv(csv_files[0])

    # Controlla se il DataFrame di base contiene la colonna "event_id"
    if not "event_id" in base_df.columns:
        raise ValueError(f"Il file CSV '{csv_files[0]}' non contiene la colonna 'event_id'.")

    # Unisce i DataFrame rimanenti al DataFrame di base
    for file in csv_files[1:]:
        df = pd.read_csv(file)
        if not "event_id" in df.columns:
            raise ValueError(f"Il file CSV '{file}' non contiene la colonna 'event_id'.")
        base_df = base_df.merge(df, on='event_id', how='outer')

    return base_df

def handle_error(error_message):
    st.error(error_message)
    st.stop()

def filter_teams(df_matches, lista_squadre):
  """
  Seleziona le righe di un dataframe Pandas in base a valori di due colonne ("home_team" e "away_team") confrontati con una lista.

  Argomenti:
    df_matches: Un dataframe Pandas contenente le colonne "home_team" e "away_team".
    lista_squadre: Una lista di stringhe contenenti i nomi delle squadre da cercare.

  Restituisce:
    Un nuovo dataframe Pandas contenente solo le righe filtrate.
  """

  # Controllo se le colonne "home_team" e "away_team" esistono
  if "home_team" not in df_matches.columns or "away_team" not in df_matches.columns:
    raise ValueError("Le colonne 'home_team' e 'away_team' non sono presenti nel dataframe.")

  # Filtro le righe in cui "home_team" o "away_team" è presente nella lista_squadre
  df_filtered = df_matches[
      (df_matches["home_team"].isin(lista_squadre)) | (df_matches["away_team"].isin(lista_squadre))
  ]

  return df_filtered


def dataframe_with_selections(df):
    """
    Seleziona il match_id da cui ricavare le informazioni.

    Argomenti:
        select: Una colonna che permette di selezionare il match_id.
        lista_squadre: Una lista di stringhe contenenti i nomi delle squadre da cercare.

    Restituisce:
        Elenco dei match_id selezionati.
    """
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=df.columns,
    )
    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)


def merge_dataframe(match_sel_id):
    """
        Recupera dati da statsbomb per ogni match_id e li unisce in un unico dataframe.

        Argomenti:
            match_sel: Una lista di match_id.

        Restituisce:
            Un dataframe contenente i dati aggregati di tutti i match_id.
    """
    df_all = None  # Inizializza dataframe cumulativo come None
    for match_id in match_sel_id:
        # Recupera i dati da statsbomb per il match_id corrente
        parser = Sbopen()

        df, related, freeze, tactics = parser.event(match_id)

        #df_events = sb.events(match_id=match_id)
        # Converte i dati in un dataframe pandas
        df_match = pd.DataFrame(df)

        # Se è il primo dataframe, inizializza il dataframe cumulativo
        if df_all is None:
            df_all = df_match.copy()
        else:
            # Unisci il dataframe del match_id corrente al dataframe cumulativo
            df_all = pd.concat([df_all, df_match], ignore_index=True)
    return df_all



import pandas as pd

def get_dataframes_for_match_ids(match_ids):
    """
    Restituisce i DataFrame per ciascun match_id.

    Args:
        match_ids (list): Lista di match_id.

    Returns:
        dict: Un dizionario in cui le chiavi sono i match_id e i valori sono i DataFrame corrispondenti.
    """
    parser = Sbopen()
    match_dataframes = {}
    for match_id in match_ids:
        df, related, freeze, tactics = parser.event(match_id)

        # Aggiungi i DataFrame al dizionario
        match_dataframes[match_id] = {
            'df': df,
            'related': related,
            'freeze': freeze,
            'tactics': tactics
        }

    return match_dataframes


def empty_fig():
    fig = go.Figure()
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=120, y1=0,
        line=dict(color="White", width=2)
    )
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=0, y1=80,
        line=dict(color="White", width=2)
    )

    fig.add_shape(
        type="line",
        x0=120, y0=0, x1=120, y1=80,
        line=dict(color="White", width=2, dash="solid")
    )

    fig.add_shape(
        type="line",
        x0=60, y0=0, x1=60, y1=80,
        line=dict(color="White", width=2, dash="solid")
    )
                
    fig.add_shape(
        type="line",
        x0=0, y0=80, x1=120, y1=80,
        line=dict(color="White", width=2, dash="solid")
    )
        
    # Imposta i limiti degli assi x e y
    fig.update_layout(
        xaxis=dict(range=[0, 120]),
        yaxis=dict(range=[0, 80]),
        title="",
        xaxis_title="Lunghezza",
        yaxis_title="Larghezza"
    )
    return fig




def passing_network(df, related, freeze, tactics, match_id, team_selected):
    # parser = Sbopen()
    # df, related, freeze, tactics = parser.event(match_id)

    for team in team_selected:
        print(team)
    
        #check for index of first sub
        sub = df.loc[df["type_name"] == "Substitution"].loc[df["team_name"] == team].iloc[0]["index"]
        #make df with successfull passes by England until the first substitution
        mask_team = (df.type_name == 'Pass') & (df.team_name == team) & (df.index < sub) & (df.outcome_name.isnull()) & (df.sub_type_name != "Throw-in")
        #taking necessary columns
        df_pass = df.loc[mask_team, ['x', 'y', 'end_x', 'end_y', "player_name", "pass_recipient_name"]]
        #adjusting that only the surname of a player is presented.
        df_pass["player_name"] = df_pass["player_name"].apply(lambda x: str(x).split()[-1])
        df_pass["pass_recipient_name"] = df_pass["pass_recipient_name"].apply(lambda x: str(x).split()[-1])


        scatter_df = pd.DataFrame()
        for i, name in enumerate(df_pass["player_name"].unique()):
            passx = df_pass.loc[df_pass["player_name"] == name]["x"].to_numpy()
            recx = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_x"].to_numpy()
            passy = df_pass.loc[df_pass["player_name"] == name]["y"].to_numpy()
            recy = df_pass.loc[df_pass["pass_recipient_name"] == name]["end_y"].to_numpy()
            scatter_df.at[i, "player_name"] = name
            #make sure that x and y location for each circle representing the player is the average of passes and receptions
            scatter_df.at[i, "x"] = np.mean(np.concatenate([passx, recx]))
            scatter_df.at[i, "y"] = np.mean(np.concatenate([passy, recy]))
            #calculate number of passes
            scatter_df.at[i, "no"] = df_pass.loc[df_pass["player_name"] == name].count().iloc[0]

        #adjust the size of a circle so that the player who made more passes 
        scatter_df['marker_size'] = (scatter_df['no'] / scatter_df['no'].max() * 1500)

        #counting passes between players
        df_pass["pair_key"] = df_pass.apply(lambda x: "_".join(sorted([x["player_name"], x["pass_recipient_name"]])), axis=1)
        lines_df = df_pass.groupby(["pair_key"]).x.count().reset_index()
        lines_df.rename({'x':'pass_count'}, axis='columns', inplace=True)
        #setting a treshold. You can try to investigate how it changes when you change it.
        lines_df = lines_df[lines_df['pass_count']>2]

        fig = empty_fig()


        # Crea i punti della passing network
        node_trace = go.Scatter(
            x=scatter_df["x"],
            y=scatter_df["y"],
            mode='markers',
            text=scatter_df["player_name"],  
            textposition="top center",
            marker=dict(size=scatter_df["marker_size"]/50, 
                    color='blue',
                    colorscale='YlGnBu',
                    line_width=2)
            )
                
        edge_traces = []
        for i, row in lines_df.iterrows():
                player1 = row["pair_key"].split("_")[0]
                player2 = row['pair_key'].split("_")[1]
                #take the average location of players to plot a line between them 
                player1_x = scatter_df.loc[scatter_df["player_name"] == player1]['x'].iloc[0]
                player1_y = scatter_df.loc[scatter_df["player_name"] == player1]['y'].iloc[0]
                player2_x = scatter_df.loc[scatter_df["player_name"] == player2]['x'].iloc[0]
                player2_y = scatter_df.loc[scatter_df["player_name"] == player2]['y'].iloc[0]
                num_passes = row["pass_count"]
                #adjust the line width so that the more passes, the wider the line
                line_width = (num_passes / lines_df['pass_count'].max() * 10)
                #plot lines on the pitch

                edge_traces.append(
                    go.Scatter(
                        x=[player1_x, player2_x],
                        y=[player1_y, player2_y],
                        mode='lines',
                        line=dict(color='gray',
                                width=line_width)
                        )
                )


        fig.add_trace(node_trace)
        fig.add_traces(edge_traces)
        fig.update_layout(title="Passing Network",
                          showlegend=False)

        # Centralità
        #calculate number of successful passes by player
        no_passes = df_pass.groupby(['player_name']).x.count().reset_index()
        no_passes.rename({'x':'pass_count'}, axis='columns', inplace=True)
        #find one who made most passes
        max_no = no_passes["pass_count"].max() 
        #calculate the denominator - 10*the total sum of passes
        denominator = 10*no_passes["pass_count"].sum() 
        #calculate the nominator
        nominator = (max_no - no_passes["pass_count"]).sum()
        #calculate the centralisation index
        centralisation_index = nominator/denominator
        # print("Centralisation index is ", centralisation_index)

    return fig, centralisation_index

def heatmap(df, team_selected, pass_type):
    for team in team_selected:
        print(team_selected)
        print(pass_type)
        
        mask_pressure = (df.team_name == team) & (df.type_name == pass_type)
        df_pressure = df.loc[mask_pressure, ['x', 'y']]
        mask_pressure = (df.team_name == team) & (df.type_name == pass_type)
        df_pass = df.loc[mask_pressure, ['x', 'y', 'end_x', 'end_y']]

        df_scaled = df_pressure.copy() 

        df_scaled['x'] = (df_scaled['x'] /df_scaled['x'].abs().max())
        df_scaled['y'] = (df_scaled['y'] /df_scaled['y'].abs().max())


        # col_1, col_2 = st.columns([1, 1])
        # with col_1:
        #     st.dataframe(df_pressure)

        # with col_2:
        #     st.dataframe(df_scaled)


        fig = empty_fig()
        fig.add_trace(go.Histogram2dContour(
                x = df_pressure['x'],
                y = df_pressure['y'],
                colorscale = 'hot',
            )
        )

        fig.update_layout(
                    title="Heatmap",
                    xaxis_title="Lunghezza",
                    yaxis_title="Ampiezza"
        )

    return fig

def passes_plot(df, team_selected, pass_type):
    mask_pressure = (df.team_name == team_selected[0]) & (df.type_name == pass_type)
    df_pass = df.loc[mask_pressure, ['x', 'y', 'end_x', 'end_y', 'outcome_name']]
    mask_complete = df_pass.outcome_name.isnull()

    # Crea la figura
    fig = empty_fig()

    # Aggiungi i passaggi completati
    for _, row in df_pass[mask_complete].iterrows():
        fig.add_trace(go.Scatter(
            x=[row['x'], row['end_x']],
            y=[row['y'], row['end_y']],
            mode='lines+markers',
            line=dict(color='#ad993c', width=2),
            marker=dict(size=5),
            name='Completed Pass'
        ))

    # Aggiungi gli altri passaggi
    for _, row in df_pass[~mask_complete].iterrows():
        fig.add_trace(go.Scatter(
            x=[row['x'], row['end_x']],
            y=[row['y'], row['end_y']],
            mode='lines+markers',
            line=dict(color='#ba4f45', width=2, dash='dash'),
            marker=dict(size=5),
            name='Other Pass'
        ))

    # Imposta layout del grafico
    fig.update_layout(
        title='Tutti i passaggi della partita',
        xaxis_title='Lunghezza',
        yaxis_title='Ampiezza',
        xaxis=dict(range=[0, 120]),
        yaxis=dict(range=[0, 80]),
        showlegend=False
    )

    return fig

def voronoi_plot(match):
    parser = Sbopen()
    frames, visible = parser.frame(match)

    frame_idx = 50
    frame_id = visible.iloc[50].id

    visible_area = np.array(visible.iloc[frame_idx].visible_area).reshape(-1, 2)
    player_position_data = frames[frames.id == frame_id]

    teammate_locs = player_position_data[player_position_data.teammate]
    opponent_locs = player_position_data[~player_position_data.teammate]

    # Calcola il diagramma di Voronoi
    points = player_position_data[['x', 'y']].values
    vor = Voronoi(points)

    # Crea la figura
    fig = empty_fig()

    # Funzione per tracciare i poligoni di Voronoi
    def plot_voronoi_polygons(vor, fig, labels, color):
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                fig.add_trace(go.Scatter(
                    x=[point[0] for point in polygon],
                    y=[point[1] for point in polygon],
                    fill="toself",
                    fillcolor=color,
                    opacity=0.4,
                    line=dict(color='white', width=3)
                ))

    # Aggiungi i poligoni di Voronoi per i compagni di squadra
    plot_voronoi_polygons(vor, fig, teammate_locs.index, 'orange')

    # Aggiungi i poligoni di Voronoi per gli avversari
    plot_voronoi_polygons(vor, fig, opponent_locs.index, 'dodgerblue')

    # Aggiungi i punti dei compagni di squadra
    fig.add_trace(go.Scatter(
        x=teammate_locs.x,
        y=teammate_locs.y,
        mode='markers',
        marker=dict(size=10, color='orange', line=dict(color='black', width=1)),
        name='Teammates'
    ))

    # Aggiungi i punti degli avversari
    fig.add_trace(go.Scatter(
        x=opponent_locs.x,
        y=opponent_locs.y,
        mode='markers',
        marker=dict(size=10, color='dodgerblue', line=dict(color='black', width=1)),
        name='Opponents'
    ))

    # Aggiungi l'area visibile
    fig.add_trace(go.Scatter(
        x=visible_area[:, 0],
        y=visible_area[:, 1],
        mode='lines',
        line=dict(color='black', dash='dash', width=2),
        name='Visible Area'
    ))

    # Imposta il layout del grafico
    fig.update_layout(
        title='Voronoi Diagram: Visible Area and Player Positions',
        xaxis_title='Lunghezza',
        yaxis_title='Ampiezza',
        xaxis=dict(range = [0, 120]),
        yaxis=dict(range = [0, 80]),
        showlegend = False
    )

    return fig

def plotting_shots(df):

    team1, team2 = df.team_name.unique()
    #A dataframe of shots
    shots = df.loc[df['type_name'] == 'Shot'].set_index('id')
    circleSize = 20
    
    fig = empty_fig()

    fig.add_trace(go.Scatter(
            x=shots.loc[shots['team_name'] == team1]['x'],
            y=shots.loc[shots['team_name'] == team1]['y'],
            mode='markers',
            marker=dict(
                        color = ['darkblue' if g else 'blue' for g in shots.loc[shots['team_name'] == team1]['outcome_name'] == 'Goal'],
                        size = [30 if g else circleSize for g in shots.loc[shots['team_name'] == team1]['outcome_name'] == 'Goal'],
                        opacity = [1 if g else 0.2 for g in shots.loc[shots['team_name'] == team1]['outcome_name'] == 'Goal']
            ),
            text=shots.loc[shots['team_name'] == team1]['player_name']
        )
    )

    # # Crea un oggetto Scatter per i tiri della squadra
    fig.add_trace(go.Scatter(
            x= 120 - shots.loc[shots['team_name'] == team2]['x'],
            y= 80 - shots.loc[shots['team_name'] == team2]['y'],
            mode='markers',
            marker=dict(
                        color = ['darkred' if g else 'red' for g in shots.loc[shots['team_name'] == team2]['outcome_name'] == 'Goal'],
                        size = [30 if g else circleSize for g in shots.loc[shots['team_name'] == team2]['outcome_name'] == 'Goal'],
                        opacity = [1 if g else 0.2 for g in shots.loc[shots['team_name'] == team2]['outcome_name'] == 'Goal']
            ),
            text=shots.loc[shots['team_name'] == team2]['player_name']
        )
    )


    title = f"Tiri di {team1} - {team2}"

    fig.update_layout(title=title,
                    showlegend = False)

    return fig


def radar_plot(df, team):
    # Seleziona le colonne per il grafico radar
    categories = ["under_pressure", "pass_shot_assist", "shot_statsbomb_xg", "aerial_won", "pass_cross", "shot_one_on_one"]
    team_data = df[df["team_name"] == team]
    m_complete = team_data.player_name.isnull()

    # # Crea il grafico radar
    st.text(df.info())
    fig = go.Figure()


    # Crea la figura
    fig = go.Figure()

    # Aggiungi i dati dei giocatori con un ciclo
    for index, row in m_complete.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[categories].values,
            theta=categories,
            fill='toself',
            name=row['player_name']
        ))

    # Configura il layout del radar plot
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 30]
            )),
        showlegend=True,
        title='Performance Comparison of Players'
    )

    return fig

def coorelation_heatmap(df, team_selected, pass_type):
    for team in team_selected:
        print(team)
        print(pass_type)
        
        df_data = df[df['type_name'].isin(pass_type)]
        df_filt= df_data[['x', 'y']]
        # df_corr = df_data.corr()
        correlation_matrix = df_data[['x', 'y']].corr()
        st.dataframe(df_data[['x', 'y']])
        st.text(correlation_matrix)
#         # Crea il grafico di correlazione
#         fig = empty_fig()
#         fig.add_trace(data=go.Splom(
#                 dimensions=[dict(label=pass_type[len(pass_type)-1],
#                                  values=df_corr['type_name']),
#                             dict(label=pass_type[len(pass_type)-1],
#                                  values=df_corr['type_name']),
# ],
#                 marker=dict(color=index_vals,
#                             showscale=False, # colors encode categorical variables
#                             line_color='white', line_width=0.5)
#                 ))
#          # Personalizza l'aspetto del grafico (titoli, assi, ecc.)
#         fig.update_layout(
#             title='Matrice di Correlazione',
#             xaxis_title='Variabili',
#             yaxis_title='Variabili'
#         )

#     return fig



        
