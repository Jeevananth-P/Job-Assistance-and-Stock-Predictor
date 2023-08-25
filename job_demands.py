import numpy as np
import streamlit as st
import plotly.express as px
from prophet import Prophet
import datetime
import pandas as pd
import plotly.graph_objects as go


def run():
    # Load CSV data
    @st.cache_resource
    def load_data():
        data = pd.read_csv("ani/Naukri Jobs Data.csv")  # Update with your CSV file path
        return data

    data = load_data()

    # Streamlit app layout
    st.title("Job Demands")

    # Display raw data
    st.subheader("Raw Data")
    st.write(data)

    # Preprocess the data
    data['Dates'] = pd.to_datetime(data['Dates'])
    filtered_data = data[data['Dates'].notnull()]

    # Get unique job roles
    job_roles = filtered_data['job_post'].unique()
    job_roles = np.insert(job_roles, 0, "")
    selected_job_role = st.selectbox("Select a job role:", job_roles)

    # Filter data for the selected job role
    job_filtered_data = filtered_data[filtered_data['job_post'] == selected_job_role]

    # Get unique companies for the selected job role
    companies = job_filtered_data['company'].unique()

    # Select multiple companies from multiselect dropdown
    selected_companies = st.multiselect("Select companies:", companies)

    for selected_company in selected_companies:
        # Filter data for the selected company
        company_data = job_filtered_data[job_filtered_data['company'] == selected_company]

        if not company_data.empty:
            # Display detailed information about the selected company
            st.subheader(f"Details for {selected_company}")
            company_info = company_data.iloc[0]
            st.info(f"Company Rating: {company_info['company_rating']}")
            st.success(f"Experience Required: {company_info['exp_required']}")
            st.warning(f"Job Location : {company_info['job_location']}")
            st.success(f"Salary:{company_info['salary_offered']}")
            s = company_info['job_description'].replace(".", "")
            st.error(f"Job Description:\n{s}")
            st.info(f"Required Skills:\n{company_info['required_skills']}")
        else:
            st.warning(f"No data available for {selected_company}.")

    st.subheader("Search for Job Openings")

    # Job openings by state
    st.subheader("Job Openings by Places")

    # Search for job openings by location
    search_location = st.text_input("Search for job openings by location:", "")

    # Filter data based on search location
    if search_location:
        filtered_data_by_location = data[data['job_location'].str.contains(search_location, case=False, na=False)]
        state_counts = filtered_data_by_location['job_location'].value_counts()

        # Display available job roles in the selected location
        available_job_roles = filtered_data_by_location['job_post'].unique()

        # Display entire raw data for job openings in the selected location and job role
        st.subheader(f"Job Openings in {search_location}")
        st.write(filtered_data_by_location)
    else:
        state_counts = data['job_location'].value_counts()

    num_categories = 10
    current_index = st.slider("Select starting category", 0, len(state_counts) - num_categories)
    selected_states = state_counts.index[current_index:current_index + num_categories]
    selected_state_counts = state_counts[selected_states]
    fig_state = px.bar(selected_state_counts, x=selected_state_counts.index, y=selected_state_counts.values)
    fig_state.update_layout(
        xaxis_title="Job Location",
        yaxis_title="Number of Openings",
        width=800,  # Adjust the width
        height=500  # Adjust the height
    )
    st.plotly_chart(fig_state)

    # Job openings by job post with slider
    st.subheader("Job Openings by Job Post")
    job_post_counts = data['job_post'].value_counts()[:]
    total_job_posts = len(job_post_counts)
    num_job_posts = 10
    job_post_index = st.slider("Select starting job post", 0, total_job_posts - num_job_posts)
    selected_job_posts = job_post_counts.index[job_post_index:job_post_index + num_job_posts]
    selected_job_post_counts = job_post_counts[selected_job_posts]
    fig_job_post = px.bar(selected_job_post_counts, x=selected_job_post_counts.index, y=selected_job_post_counts.values)
    fig_job_post.update_layout(
        xaxis_title="Job Post",
        yaxis_title="Number of Openings",
        width=800,  # Adjust the width
        height=500  # Adjust the height
    )
    st.plotly_chart(fig_job_post)

    company_name = st.text_input("Enter a company name to filter job roles:", "")

    if company_name:
        roles_in_company = data[data['company'] == company_name]['job_post'].unique()
        st.write(f"Job Roles in {company_name}:")
        st.write(roles_in_company)

    # Predict Future Openings
    st.subheader("Predicted Openings Based on Trends")

    # Preprocess the data
    data['Dates'] = pd.to_datetime(data['Dates'])
    filtered_data = data[data['Dates'].notnull()]

    # Get unique job roles
    job_roles = filtered_data['job_post'].unique()

    # Select job role from dropdown
    selected_job_role = st.selectbox("Select a job role:", job_roles)

    # Filter data for the selected job role
    job_filtered_data = filtered_data[filtered_data['job_post'] == selected_job_role]

    # Group data by Date and count openings
    openings_by_date = job_filtered_data.groupby('Dates').size().reset_index(name='Openings')

    # Check if the dataset has enough rows for modeling
    if len(openings_by_date) >= 2:
        # Create a time series model for predictions
        prophet_data = openings_by_date.rename(columns={'Dates': 'ds', 'Openings': 'y'})
        model = Prophet(weekly_seasonality=True)  # Add weekly seasonality
        model.fit(prophet_data)

        # Create future dates for prediction based on slider value
        # num_weeks = st.slider("Select number of weeks to predict", 1, 4, 1
        num_weeks = 1
        future_dates = pd.date_range(start=prophet_data['ds'].max(), periods=num_weeks * 7, freq='D')
        future_dates_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_dates_df)

        # Create a combined graph
        combined_fig = go.Figure()

        # Add historical data trace
        historical_trace = go.Scatter(
            x=openings_by_date['Dates'],
            y=openings_by_date['Openings'],
            mode='lines+markers',
            name='Historical Openings',
            line=dict(color='blue')
        )
        combined_fig.add_trace(historical_trace)

        # Add predicted data trace
        predicted_trace = go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines+markers',
            name='Predicted Openings',
            line=dict(color='red')
        )
        combined_fig.add_trace(predicted_trace)

        # Adjust x-axis range for better alignment
        combined_fig.update_xaxes(range=[openings_by_date['Dates'].min(), forecast['ds'].max()])

        # Set y-axis tick format for predicted data trace
        combined_fig.update_yaxes(tickformat=',.0f', title='Openings')

        # Display the combined graph
        st.plotly_chart(combined_fig)

    else:
        st.write(f"Not enough data for {selected_job_role} to make predictions.")


if __name__ == "__main__":
    run()
