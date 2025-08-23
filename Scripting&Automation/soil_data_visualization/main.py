import streamlit as st
import pandas as pd
import altair as alt

# Set the page configuration
st.set_page_config(
    page_title="Soil Data Visualization",
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

    # Clean up column names by stripping leading/trailing whitespace
    combined_df.columns = combined_df.columns.str.strip().str.rstrip(':')

    # Clean up the 'Analyte' column values by stripping whitespace
    combined_df['Analyte'] = combined_df['Analyte'].str.strip()

    # Convert 'Result' and 'Optimal' columns to numeric, coercing errors to NaN
    # This handles any non-numeric values gracefully
    combined_df['Result'] = pd.to_numeric(combined_df['Result'], errors='coerce')
    combined_df['Optimal'] = pd.to_numeric(combined_df['Optimal'], errors='coerce')

    return combined_df

df = load_data()

if df is not None:
    # --- App Layout and Interactive Components ---
    st.title("Soil Data Dashboard")
    st.markdown(
        """
        This application visualizes soil data from multiple sheets. Use the dropdown
        menu to select an `Analyte` and see a comparison of the measured `Result`
        and the `Optimal` value for each grower.
        """
    )

    # Create a list of unique analytes for the dropdown menu
    unique_analytes = df['Analyte'].unique()
    selected_analyte = st.selectbox(
        "Select an Analyte:",
        unique_analytes
    )

    # Filter the DataFrame based on the user's selection
    analyte_df = df[df['Analyte'] == selected_analyte]

    if not analyte_df.empty:
        # Melt the DataFrame to create a long-form dataset suitable for Altair
        # This allows us to plot both 'Result' and 'Optimal' side-by-side
        melted_df = analyte_df.melt(
            id_vars=['Grower'],
            value_vars=['Result', 'Optimal'],
            var_name='Metric',
            value_name='Value'
        )

        # Create the Altair chart
        chart = alt.Chart(melted_df).mark_bar().encode(
            # X-axis for 'Grower', with a tooltip to show the grower name
            x=alt.X('Grower:N', axis=None),
            # Y-axis for the value, with a clear title
            y=alt.Y('Value:Q', title=f'{selected_analyte} Value'),
            # Use color to differentiate between Result and Optimal
            color=alt.Color('Metric:N', legend=alt.Legend(title="Metric")),
            # Add a column to separate bars for each Grower
            column=alt.Column('Grower:N', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
            # Add tooltips for interactive hovering
            tooltip=['Grower', 'Metric', 'Value']
        ).properties(
            title=f'{selected_analyte} Result vs. Optimal Value by Grower',
            width=200,
        ).configure_axis(
            grid=False
        ).configure_header(
            titleOrient="bottom",
            labelOrient="bottom",
            labelPadding=5,
            titlePadding=5
        )

        # Display the chart
        st.altair_chart(chart, use_container_width=False)

    else:
        st.warning("No data available for the selected analyte.")

    # Optional: Display the raw data for inspection
    st.subheader("Raw Data")
    st.dataframe(df)

