import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import datetime
import streamlit.components.v1 as components

# Configure the page
st.set_page_config(
    page_title="Basketball ML Rankings",
    page_icon=":basketball:",
    layout="wide"
)

def load_rankings(filepath='playAll_avgRankings.csv'):
    """Load and prepare basketball rankings data."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        'Team': 'Team',
        'Conf': 'Conference',
        'overall_avg_spread': 'Rating',
        'overall_avg_win': 'Win%'
    })
    df = df.drop(columns=['home_team_avg_spread', 'away_team_avg_spread', 'home_team_avg_win', 'away_team_avg_win'])
    df['Conference'] = df['Conference'].astype(str)
    df['Rating'] = df['Rating'] * -1
    df = df.sort_values('Rating', ascending=False).reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    return df

def load_predictions(filepath='individual_predictions.csv'):
    """Load and prepare matchup prediction data."""
    df = pd.read_csv(filepath)
    return df

def load_team_logos(filepath='team_logos.csv'):
    """Load team logos data."""
    df = pd.read_csv(filepath)
    return df

def create_conference_summary(df):
    """Create conference-level summary statistics."""
    conf_summary = df.groupby('Conference').agg({
        'Team': 'count',
        'Rating': 'mean',
        'Win%': 'mean',
    }).round(2)
    conf_summary = conf_summary.rename(columns={'Team': 'Teams'})
    conf_summary = conf_summary.sort_values('Rating', ascending=False)
    conf_summary = conf_summary.reset_index()
    return conf_summary

def create_style_functions():
    """Create styling functions for the DataFrame display."""
    def background_gradient(s, m, M, cmap='BrBG'):
        rng = M - m
        if rng == 0:
            rng = 1
        norm = lambda x: (x - m) / rng
        normed = s.apply(norm)
        colors = plt.cm.BrBG(normed)
        def get_text_color(rgb):
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            return 'white' if luminance < 0.75 else 'black'
        return [
            f'background-color: {matplotlib.colors.rgb2hex(x)}; '
            f'color: {get_text_color(x)}; '
            f'text-align: center !important;'
            for x in colors
        ]
    
    def style_df(df):
        styled_df = df.style
        for col in ['Rating', 'Win%']:
            styled_df = styled_df.apply(
                background_gradient,
                m=df[col].min(),
                M=df[col].max(),
                subset=[col]
            )
        styled_df = styled_df.format({
            'Rank': '{}',
            'Rating': '{:.2f}',
            'Win%': '{:.3f}',
        })
        return styled_df
    
    return style_df

def setup_page():
    """Configure page layout and add explanatory text."""
    st.title("NCAA Basketball Power Rankings")
    st.subheader("Powered by Machine Learning")
    st.markdown("""
    The machine learning model predicts the margin of victory for
     every possible NCAA Division 1 basketball matchup, both home
     and away. Each team's rating reflects their average predicted
     margin of victory across all matchups. Each team's win percentage 
    represents their predicted win percentage across all matchups.
    """)

def display_matchup_predictor(predictions_df, team_logos_df):
    """Display the matchup predictor section with side-by-side team dropdowns and full-width cards."""
    st.subheader("Matchup Predictor")

    # Define CSS styles for cards and text
    st.markdown(
        """
        <style>
        .prediction-card {
            background-color: #262730;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #444;
            margin-bottom: 15px;
            color: #e0e0e0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Team selection dropdowns side by side
    teams = sorted(predictions_df['Home_Team'].unique())
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, index=teams.index("Pittsburgh"))
    with col2:
        team2 = st.selectbox("Team 2", teams, index=teams.index("West Virginia"))

    # Validate team selection
    if team1 == team2:
        st.warning("Please select two different teams.")
    else:
        # Retrieve spreads
        S_home = predictions_df[(predictions_df['Home_Team'] == team1) & (predictions_df['Away_Team'] == team2)]['Average_Spread'].values[0]
        S_away = predictions_df[(predictions_df['Home_Team'] == team2) & (predictions_df['Away_Team'] == team1)]['Average_Spread'].values[0]
        neutral_spread = (S_home - S_away) / 2

        # Helper function to determine favored team and points
        def get_prediction(spread, team_home, team_away):
            if spread < 0:
                favorite = team_home
                points = -spread
            else:
                favorite = team_away
                points = spread
            return favorite, points

        # Calculate predictions
        favorite_home, points_home = get_prediction(S_home, team1, team2)
        favorite_away, points_away = get_prediction(S_away, team2, team1)
        favorite_neutral, points_neutral = (
            (team1, -neutral_spread) if neutral_spread < 0 else (team2, neutral_spread)
        )

        # Fetch logos
        favored_home_logo = team_logos_df.loc[team_logos_df['Team'] == favorite_home, 'Logo'].values[0]
        favored_away_logo = team_logos_df.loc[team_logos_df['Team'] == favorite_away, 'Logo'].values[0]
        favored_neutral_logo = team_logos_df.loc[team_logos_df['Team'] == favorite_neutral, 'Logo'].values[0]

        # CSS classes for favored team text
        favored_class_home = 'team1-favored' if favorite_home == team1 else 'team2-favored'
        favored_class_away = 'team1-favored' if favorite_away == team1 else 'team2-favored'
        favored_class_neutral = 'team1-favored' if favorite_neutral == team1 else 'team2-favored'

        # Helper function for card HTML
        def prediction_card(scenario, favorite, points, favored_logo, favored_class):
            return f"""
            <div class="prediction-card">
                <h4 style="color: #e0e0e0;">{scenario}</h4>
                <p>
                    <img src="{favored_logo}" width="60" style="vertical-align:middle; margin-right:10px;">
                    <span class="{favored_class}">{favorite} by {points:.1f} points</span>
                </p>
            </div>
            """

        # Display predictions
        st.markdown(prediction_card(f"{team1} at Home", favorite_home, points_home, favored_home_logo, favored_class_home), unsafe_allow_html=True)
        st.markdown(prediction_card(f"{team2} at Home", favorite_away, points_away, favored_away_logo, favored_class_away), unsafe_allow_html=True)
        st.markdown(prediction_card("Neutral Court", favorite_neutral, points_neutral, favored_neutral_logo, favored_class_neutral), unsafe_allow_html=True)

def main():
    """Main application logic."""
    # Inject custom CSS for table alignment and reduced subheader margin
    st.markdown("""
    <style>
        .stDataFrame td {
            text-align: center !important;
        }
        .stDataFrame td:nth-child(2) {
            text-align: left !important;
        }
        /* Add padding between columns */
        [data-testid="column"] {
            padding: 0 10px;
        }
        /* Reduce space below subheaders */
        h3 {
            margin-bottom: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Load data
    rankings_df = load_rankings()
    predictions_df = load_predictions()
    team_logos_df = load_team_logos()
    
    # Setup page layout
    setup_page()
    
    # Display timestamp left-aligned below main headings
    timestamp = os.path.getmtime('playAll_avgRankings.csv')
    file_date = datetime.datetime.fromtimestamp(timestamp).strftime("%B %d, %Y")
    st.markdown(
        f'<p style="text-align: left; color: gray; font-size: 12px;">Updated: {file_date}</p>',
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Create three main columns
    main_col1, main_col2, main_col3 = st.columns(3)
    
    # First column: Matchup Predictor
    with main_col1:
        display_matchup_predictor(predictions_df, team_logos_df)
    
    # Second column: Team Rankings
    with main_col2:
        st.subheader("Team Rankings")

        st.markdown(
            """
            <style>
            .prediction-card {
                background-color: #262730;
                padding: 10px;
                border-radius: 10px;
                border: 1px solid #444;
                margin-bottom: 15px;
                color: #e0e0e0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Get selected conferences from session state
        selected_conferences = st.session_state.get("conference_filter", [])
        # Filter rankings_df based on selected conferences
        if selected_conferences:
            filtered_df = rankings_df[rankings_df['Conference'].isin(selected_conferences)]
        else:
            filtered_df = rankings_df.copy()
        # Team filter options are teams in the filtered_df
        team_options = sorted(filtered_df['Team'].unique())
        # Team filter
        selected_teams = st.multiselect(
            "Filter teams",
            options=team_options,
            placeholder="Select teams",
            key="team_filter"
        )
        # Further filter based on selected teams
        if selected_teams:
            filtered_df = filtered_df[filtered_df['Team'].isin(selected_teams)]
        # Display the table
        style_df = create_style_functions()
        styled_table = style_df(filtered_df)
        st.dataframe(
            styled_table,
            hide_index=True,
            use_container_width=True,
            height=910
        )
        # Download button
        csv = filtered_df.to_csv().encode('utf-8')
        st.download_button(
            "Download Team Rankings",
            csv,
            "team_rankings.csv",
            "text/csv",
            key='download-csv'
        )
    
    # Third column: Conference Rankings
    with main_col3:
        st.subheader("Conference Rankings")

        st.markdown(
            """
            <style>
            .prediction-card {
                background-color: #262730;
                padding: 10px;
                border-radius: 10px;
                border: 1px solid #444;
                margin-bottom: 15px;
                color: #e0e0e0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Conference filter within Conference Rankings section
        selected_conferences = st.multiselect(
            "Filter conferences",
            options=sorted(rankings_df['Conference'].unique()),
            placeholder="Select conferences",
            key="conference_filter"
        )
        # Create and filter conference summary
        conf_summary = create_conference_summary(rankings_df)
        if selected_conferences:
            conf_summary = conf_summary[conf_summary['Conference'].isin(selected_conferences)]
        # Style and display table
        def style_conf_summary(df):
            return df.style.format({
                'Teams': '{}',
                'Rating': '{:.2f}',
                'Win%': '{:.3f}',
            }).background_gradient(
                subset=['Rating', 'Win%'],
                cmap='BrBG'
            )
        styled_conf_summary = style_conf_summary(conf_summary)
        st.dataframe(
            styled_conf_summary,
            hide_index=True,
            use_container_width=True,
            height=910
        )
        # Download button
        conf_csv = conf_summary.to_csv().encode('utf-8')
        st.download_button(
            "Download Conference Rankings",
            conf_csv,
            "conference_rankings.csv",
            "text/csv",
            key='download-conf-csv'
        )
    
    # Add footer
    #st.markdown("---")
    # st.markdown(
    #     """
    #     <div style="
    #         position: relative;
    #         bottom: 0;
    #         width: 100%;
    #         text-align: center;
    #         padding: 20px 0;
    #         color: #666;
    #         font-size: 14px;
    #     ">
    #         <p>Created by Trevor Barr</p>
    #         <p>
    #             <a href="https://twitter.com/YourHandle" target="_blank" 
    #                style="margin: 0 10px; color: #0077B5; text-decoration: none;">
    #                 Twitter
    #             </a>
    #             •
    #             <a href="https://github.com/YourHandle" target="_blank" 
    #                style="margin: 0 10px; color: #0077B5; text-decoration: none;">
    #                 GitHub
    #             </a>
    #             •
    #             <a href="https://www.linkedin.com/in/trevorabarr/" target="_blank" 
    #                style="margin: 0 10px; color: #0077B5; text-decoration: none;">
    #                 LinkedIn
    #             </a>
    #         </p>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )

if __name__ == "__main__":
    main()