import streamlit as st
import pandas as pd
import time 
import numpy as np
from process import *
import plotly.graph_objects as go
from statsbombpy import sb
import streamlit.components.v1 as components
import networkx as nx
from mplsoccer import Pitch, Sbopen
import altair as alt




BASE_DIR = "./data/"

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
# components.html(
#     """
#         <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.2/dist/umd/popper.min.js" integrity="sha384-IQsoLXl5PILFhosVNubq5LC7Qb9DXgDA9i+tQ8Zj3iwWAwPtgFTxbJ8NT4GN1R8p" crossorigin="anonymous"></script>
#         <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.min.js" integrity="sha384-cVKIPhGWiC2Al4u+LWgxfKTRIcfu0JTxR+EQDz/bgldoEyl4H0zUF0QKbrJ0EcQF" crossorigin="anonymous"></script>
#     """
#     )
st.title("Data Analyst EFC")

st.subheader("I dati dell'analisi provengono dal sito [statsbomb](https://www.statsbomb.com/).")

st.html("<hr>")
st.text("Campionato Italiano Serie A, stagione 2015/16.")
st.html("<hr>")

st.subheader("Teams")
df_matches = sb.matches(competition_id=12, season_id=27)
# df_matches.to_csv(BASE_DIR + 'matches.csv', index=False)

teams = df_matches['home_team'].unique()
team_selected = st.multiselect("Seleziona la squadra da studiare:", sorted(teams))
st.text(team_selected)
if len(team_selected) != 0 and len(team_selected) <= 1:
    df_teams = filter_teams(df_matches, team_selected)
    cols = ["match_id", "home_team", "away_team", "home_score", "away_score", "match_week"]
    df_cols = df_teams[["match_id", "home_team", "away_team", "home_score", "away_score", "match_week"]]
    # st.dataframe(df_cols.sort_values(by='match_week'))
    st.html("<hr>")

    n_matches = max(df_matches['match_week'])
    st.text("Seleziona l'intervallo delle giornate da analizzre")
    matches = st.slider("", 1, n_matches, (1, 3))
    st.write("Intervallo selezionato: dalla ", matches[0], "alla ", matches[1], " giornata.")

    match_id = df_cols[(df_cols['match_week'] >= matches[0]) & (df_cols['match_week'] <= matches[1])].sort_values(by='match_week')

    all_data = st.checkbox("Vuoi analizzare l'intero intervallo di giornate?", key="disabled")


    selection = dataframe_with_selections(match_id)
    # st.write("Data selected:")

    if all_data==False:

        try:
            # st.table(selection)
            # match = selection['match_id'].iloc[0]
            # df_events = sb.events(match_id=match)
            # st.write(match)
            match_sel = selection['match_id'].values.tolist()
            # st.write(selection.values.tolist())

            # df_data = pd.DataFrame()
            # match_sel = selection['match_id'].values.tolist()
            if len(match_sel) > 0:
                #df_all = merge_dataframe(match_sel)

                match_data = get_dataframes_for_match_ids(match_sel)

            #     st.write(match_sel)
            #     for match_id in match_sel:
            #         st.text(match_id)
            #         # Recupera i dati da statsbomb per il match_id corrente
            #         df_events = sb.events(match_id=match_id)
            #         # Unisci il dataframe del match_id corrente al dataframe cumulativo
            #         df_data.append(df_events, ignore_index=True)

                # df_event = df_all[[ "match_id", "minute", "pass_body_part",  "pass_height", 
                # "pass_length", "pass_outcome", "pass_recipient", "pass_type",
                # "period", "player", "position", "possession", "possession_team", "second",
                # "team", "type", "under_pressure", "location", "pass_end_location"]]

                # st.dataframe(df_event[df_event['team'].isin(team_selected)])

                # creazione della passing network per ogni giornata
                for match in match_sel:

                    df = match_data[match]['df']
                    print(type(df.describe()))
                    related = match_data[match]['related']
                    freeze = match_data[match]['freeze']
                    tactics = match_data[match]['tactics']

                    # pass_network = df_event[(df_event['match_id']==match) & (df_event['team'].isin(team_selected))]
                    info_match = selection[selection['match_id']==match]

                    for row in info_match.itertuples():
                        print(row)
                        index, match_id, squadra_casa, squadra_ospite, gol_casa, gol_ospite, giornata = row
                        data_str = f"Match ID: {match_id} | Giornata: {giornata} | {squadra_casa} - {squadra_ospite} {gol_casa} - {gol_ospite}"

                        st.info(data_str)

                    fig, centrality = passing_network(df, related, freeze, tactics, match, team_selected)
                    st.plotly_chart(fig, use_container_width=True)

                    centralisation_index = f"Indice di centralità: {round(centrality*100, 2)}%"
                    st.markdown(centralisation_index)

                    type_name = ['Pass', 'Pressure', 'Duel', 'Foul Committed']

                    df_filtered = df[df['type_name'].isin(type_name)]
                    selected_pass_type = st.multiselect("Seleziona il tipo di evento", df_filtered['type_name'].unique())

                    #selected_pass_type = st.multiselect("Seleziona il tipo di evento", df['type_name'].isin(type_name)])
                    if len(selected_pass_type) != 0 and len(selected_pass_type) <= 1:
                        st.text(selected_pass_type[0])
                        st.plotly_chart(heatmap(df, team_selected, selected_pass_type[0]))
                        # radar plot
                        # df_radar = df[["match_id", "team_name", "under_pressure", "pass_shot_assist", "shot_statsbomb_xg", "aerial_won", "pass_cross", "shot_one_on_one"]]
                        # st.plotly_chart(radar_plot(df, team_selected[0]))
                        if selected_pass_type[0] == 'Pass':
                            st.plotly_chart(passes_plot(df, team_selected, selected_pass_type[0]))
                        # Voronoi
                        # st.plotly_chart(voronoi_plot(match=3788745))
                        st.plotly_chart(plotting_shots(df))
                    elif len(selected_pass_type) > 1:
                        pass
                    else:
                        pass
                
                    st.html("<hr>")
        except IndexError:
            match = None  
            st.write(match)          

        with st.spinner("Elaborazione dati..."):
            time.sleep(5)
        st.success("Fatto!")
    else:
        st.text(all_data)
        match_ids = []
        for i in range(matches[0], matches[1] + 1):
            match = df_cols[df_cols['match_week']==i]['match_id'].to_list()
            match_ids.append(match[0])

        df = merge_dataframe(match_ids)
        # df = df_total['df']
            # print(type(df))
            # related = match_data[match]['related']
            # freeze = match_data[match]['freeze']
            # tactics = match_data[match]['tactics']
        
        type_name = ['Pass', 'Pressure', 'Duel', 'Foul Committed']

        df_filtered = df[df['type_name'].isin(type_name)]
        selected_pass_type = st.multiselect("Seleziona il tipo di evento da visualizzare per l'intervallo di giornato selezionato:", df_filtered['type_name'].unique())

        #selected_pass_type = st.multiselect("Seleziona il tipo di evento", df['type_name'].isin(type_name)])
        if len(selected_pass_type) != 0 and len(selected_pass_type) <= 1:
            st.text(selected_pass_type[0])
            st.text(team_selected)
            st.plotly_chart(heatmap(df,team_selected, selected_pass_type[0]))
            # st.plotly_chart(radar_plot(df, team_selected[0]))

        elif len(selected_pass_type) > 1:
            st.text("Puoi selezionare solo un paramtero, per adesso!")
            #st.plotly_chart(coorelation_heatmap(df,team_selected, selected_pass_type))
            # coorelation_heatmap(df,team_selected, selected_pass_type)
        else:
            st.text("Puoi selezionare solo un paramtero, per adesso!")


        
 
elif len(team_selected) > 1:
    st.text("Puoi selezionare solo una squadra!")
else:
    st.text("Non hai selezionato nessuna squadra!")





# with st.sidebar:
#     files = []
#     uploaded_files = st.file_uploader("Carica i file CSV da elaborare", accept_multiple_files=True)

#     for uploaded_file in uploaded_files:
#         bytes_data = uploaded_file.read()
#         files.append(BASE_DIR + uploaded_file.name)

#     # with st.spinner("Loading..."):
#     #     time.sleep(5)
#     # st.success("Done!")

# if files:
#     try:
#         events = merge_csv(files)
#         st.success("Elaborazione terminata!")

#         # col1, col2 = st.columns([1, 1])

#         # with col1:
#         #     st.subheader("FILTRI")
#         #     st.data_editor(
#         #         events.columns.to_frame().reset_index(drop=True),  # Create a temporary DataFrame with column names
#         #         column_config={
#         #             "index": st.column_config.TextColumn(  # Configure the only column
#         #                 "Colonne da filtrare",  # Remove the default label
#         #             )
#         #         },
#         #         hide_index=True,  # Hide the index of the temporary DataFrame
#         #         num_rows="dynamic",
#         #     )

#         # with col2:
#         #     col2.subheader("COLONNE")
#         #     st.data_editor(
#         #         events.columns.to_frame().reset_index(drop=True),  # Create a temporary DataFrame with column names
#         #         column_config={
#         #             "index": st.column_config.TextColumn(  # Configure the only column
#         #                 "Colonne da visualizzare",  # Remove the default label
#         #             )
#         #         },
#         #         hide_index=True,  # Hide the index of the temporary DataFrame
#         #         num_rows="dynamic",
#         #     )        

#         # col3, col4 = st.columns([1, 1])

#         # with col3:
#         #     col3.subheader("RIGHE")
#         #     st.data_editor(
#         #         events.columns.to_frame().reset_index(drop=True),  # Create a temporary DataFrame with column names
#         #         column_config={
#         #             "index": st.column_config.TextColumn(  # Configure the only column
#         #                 "Righe da visualizzare",  # Remove the default label
#         #             )
#         #         },
#         #         hide_index=True,  # Hide the index of the temporary DataFrame
#         #         num_rows="dynamic",
#         #     )

#         # with col4:
#         #     col4.subheader("VALORI")
#         #     st.data_editor(
#         #         events.columns.to_frame().reset_index(drop=True),  # Create a temporary DataFrame with column names
#         #         column_config={
#         #             "index": st.column_config.TextColumn(  # Configure the only column
#         #                 "Valori da calcolare",  # Remove the default label
#         #             )
#         #         },
#         #         hide_index=True,  # Hide the index of the temporary DataFrame
#         #         num_rows="dynamic",
#         #     )             

#         # Raggruppa per 'gruppo' e calcola la somma dei 'valore'
#         data_group = events.groupby('team_name', as_index=False).agg(max_value=('pass_length', 'max'), min_value=('pass_length', 'min'))

#         st.dataframe(data_group)
     



#     except ValueError as e:
#         handle_error(e.args[0])
