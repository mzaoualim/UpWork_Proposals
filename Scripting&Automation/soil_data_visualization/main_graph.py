import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import altair as alt

# Set the page configuration
st.set_page_config(
    page_title="Soil Data Network Graph",
    layout="wide",
)

# --- Data Loading and Cleaning ---

@st.cache_data
def load_data():
    """
    Loads all relevant CSV data, combines them into a single DataFrame,
    and performs initial data cleaning.
    """
    # List of file names for the four sheets
    sheets = [
        'Sheet1', 'Sheet2', 'Sheet3', 'Sheet4'
    ]

    all_data = []
    # Loop through each file and append its DataFrame to a list
    for file in sheets:
        try:
            df = pd.read_excel('Sample-Data-For-Graphs.xlsx', sheet_name=file)
            all_data.append(df)
        except FileNotFoundError:
            st.error(f"File not found: {file}. Please make sure it's in the same directory.")
            return None

    # Concatenate all DataFrames into one
    combined_df = pd.concat(all_data, ignore_index=True)

    # Clean up column names by stripping leading/trailing whitespace and the colon
    combined_df.columns = combined_df.columns.str.strip().str.rstrip(':')

    # Clean up the 'Analyte' column values by stripping whitespace
    combined_df['Analyte'] = combined_df['Analyte'].str.strip()

    # Convert 'Result' and 'Optimal' columns to numeric, coercing errors to NaN
    combined_df['Result'] = pd.to_numeric(combined_df['Result'], errors='coerce')
    combined_df['Optimal'] = pd.to_numeric(combined_df['Optimal'], errors='coerce')

    return combined_df

# --- App Layout and Interactive Components ---
st.title("Soil Data Relationship Map")
st.markdown(
    """
    This application visualizes the relationships between growers, their samples,
    and the analytes tested.
    
    Select a **Grower** from the dropdown below to see their specific network of analytes and a comparison of their results against optimal values.
    """
)

df = load_data()

if df is not None:
    # Create a list of unique growers for the dropdown menu
    unique_growers = df['Grower'].unique()
    selected_grower = st.selectbox(
        "Select a Grower:",
        unique_growers
    )
    
    # Filter the DataFrame for the selected grower
    grower_df = df[df['Grower'] == selected_grower].copy()
    
    if not grower_df.empty:
        st.subheader(f"Network Graph for {selected_grower}")
        
        # Initialize a NetworkX graph
        G = nx.Graph()
        
        # Define colors for different types of nodes
        node_colors = {
            'Grower': '#4CAF50',  # Green
            'Analyte': '#FFC107'    # Yellow
        }

        # Add the grower node as the central hub
        G.add_node(selected_grower, size=20, color=node_colors['Grower'], title=f'Grower: {selected_grower}')

        # Add analyte nodes and edges to the central grower node
        for _, row in grower_df.iterrows():
            analyte = row['Analyte']
            result = row['Result']
            
            # Add analyte node
            G.add_node(analyte, size=10, color=node_colors['Analyte'], title=f'Analyte: {analyte}\nResult: {result}')
            
            # Add an edge from the grower to the analyte
            G.add_edge(selected_grower, analyte)
            
        # Convert NetworkX graph to a Pyvis network
        net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='local', notebook=False)
        net.from_nx(G)
        
        # Add a legend
        st.markdown("### Graph Legend")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f'<div style="background-color:{node_colors["Grower"]}; color:white; padding: 5px; border-radius: 5px; text-align: center;">Grower</div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div style="background-color:{node_colors["Analyte"]}; color:black; padding: 5px; border-radius: 5px; text-align: center;">Analyte</div>', unsafe_allow_html=True)

        # Save the Pyvis network to an HTML file
        net.save_graph('interactive_graph.html')
        
        # Read the saved HTML file and display it in Streamlit
        try:
            HtmlFile = open("interactive_graph.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.html(source_code, height=500)
        except FileNotFoundError:
            st.error("The network graph could not be generated.")

        st.subheader(f"Analyte Values for {selected_grower}")
        
        # Melt the DataFrame to create a long-form dataset suitable for Altair
        melted_df = grower_df.melt(
            id_vars=['Analyte'],
            value_vars=['Result', 'Optimal'],
            var_name='Metric',
            value_name='Value'
        )

        # Create the Altair chart
        chart = alt.Chart(melted_df).mark_bar().encode(
            x=alt.X('Analyte:N', axis=alt.Axis(labels=True, title='Analyte', labelAngle=-45)),
            y=alt.Y('Value:Q', title=f'Value'),
            color=alt.Color('Metric:N', legend=alt.Legend(title="Metric")),
            tooltip=['Analyte', 'Metric', 'Value']
        ).properties(
            title=f'Result vs. Optimal Value for {selected_grower}'
        ).configure_axis(
            grid=False
        ).configure_header(
            titleOrient="bottom",
            labelOrient="bottom",
            labelPadding=5,
            titlePadding=5
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

    else:
        st.warning("No data available for the selected grower.")

    # Display the raw data for inspection
    st.subheader("Raw Data")
    st.dataframe(df)

