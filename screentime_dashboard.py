"""
Indian Kids Screen Time Analysis Dashboard (2025)
Author: Data Visualization Lab Assignment
Description: Interactive dashboard for analyzing screen time patterns among Indian children
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Indian Kids Screen Time Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING (UPDATED) ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        color: #000000; /* Added: Forces black text for visibility */
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f5e9;
        color: #333333; /* Added: Forces dark grey text for visibility */
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING FUNCTION ====================
@st.cache_data
def load_data(uploaded_file=None):
    """
    Load and preprocess the screen time dataset
    Handles missing values and data type conversions
    """
    try:
        if uploaded_file is not None:
            # Load data from uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Create Age_Group from Age column
            df['Age_Group'] = pd.cut(df['Age'], 
                                     bins=[0, 10, 13, 16, 20], 
                                     labels=['8-10 years', '11-13 years', '14-16 years', '17-18 years'])
            
            # Rename columns for consistency
            df = df.rename(columns={
                'Avg_Daily_Screen_Time_hr': 'Screen_Time_Hours',
                'Primary_Device': 'Device_Type',
                'Urban_or_Rural': 'Location'
            })
            
        else:
            # Create sample data for demonstration
            st.warning("⚠️ No dataset uploaded. Using sample data for demonstration.")
            df = create_sample_data()
        
        # Data Cleaning
        # 1. Handle missing values
        df = df.dropna(subset=['Age', 'Screen_Time_Hours'])  # Drop rows with critical missing values
        df['Gender'] = df['Gender'].fillna('Unknown')
        df['Device_Type'] = df['Device_Type'].fillna('Unknown')
        df['Location'] = df['Location'].fillna('Unknown')
        
        # Fill Health_Impacts and other columns if they exist
        if 'Health_Impacts' in df.columns:
            df['Health_Impacts'] = df['Health_Impacts'].fillna('None Reported')
        
        if 'Educational_to_Recreational_Ratio' in df.columns:
            df['Educational_to_Recreational_Ratio'] = pd.to_numeric(
                df['Educational_to_Recreational_Ratio'], errors='coerce'
            ).fillna(0)
        
        if 'Exceeded_Recommended_Limit' in df.columns:
            df['Exceeded_Recommended_Limit'] = df['Exceeded_Recommended_Limit'].fillna(False)
        
        # 2. Convert data types
        df['Screen_Time_Hours'] = pd.to_numeric(df['Screen_Time_Hours'], errors='coerce')
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        
        # 3. Remove duplicates
        df = df.drop_duplicates()
        
        # 4. Data validation - remove outliers (screen time > 24 hours)
        df = df[df['Screen_Time_Hours'] <= 24]
        df = df[df['Screen_Time_Hours'] >= 0]
        
        # 5. Ensure Age_Group exists
        if 'Age_Group' not in df.columns:
            df['Age_Group'] = pd.cut(df['Age'], 
                                     bins=[0, 10, 13, 16, 20], 
                                     labels=['8-10 years', '11-13 years', '14-16 years', '17-18 years'])
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ==================== SAMPLE DATA GENERATOR ====================
def create_sample_data():
    """
    Creates sample data for demonstration purposes
    """
    np.random.seed(42)
    n_samples = 1000
    
    genders = ['Male', 'Female']
    devices = ['Smartphone', 'Tablet', 'Laptop', 'Desktop', 'Smart TV']
    health_impacts = ['Eye Strain', 'Sleep Disruption', 'Posture Issues', 'None Reported', 'Multiple Issues']
    locations = ['Urban', 'Rural']
    
    # Generate ages and create age groups
    ages = np.random.randint(8, 19, n_samples)
    
    data = {
        'Age': ages,
        'Gender': np.random.choice(genders, n_samples),
        'Device_Type': np.random.choice(devices, n_samples),
        'Health_Impacts': np.random.choice(health_impacts, n_samples),
        'Location': np.random.choice(locations, n_samples),
        'Screen_Time_Hours': np.random.exponential(3, n_samples) + np.random.uniform(1, 3, n_samples),
        'Educational_to_Recreational_Ratio': np.random.uniform(0.1, 0.8, n_samples),
        'Exceeded_Recommended_Limit': np.random.choice([True, False], n_samples, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Create Age_Group
    df['Age_Group'] = pd.cut(df['Age'], 
                             bins=[0, 10, 13, 16, 20], 
                             labels=['8-10 years', '11-13 years', '14-16 years', '17-18 years'])
    
    # Add some realistic patterns
    df.loc[df['Age_Group'] == '17-18 years', 'Screen_Time_Hours'] *= 1.3
    df.loc[df['Location'] == 'Urban', 'Screen_Time_Hours'] *= 1.15
    df['Screen_Time_Hours'] = df['Screen_Time_Hours'].clip(upper=15)
    
    return df

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.markdown('<div class="main-header">📱 Indian Kids Screen Time Analysis Dashboard 2025</div>', 
                unsafe_allow_html=True)
    
    # Sidebar for file upload and filters
    st.sidebar.header("📊 Dashboard Controls")
    
    # File uploader
    st.sidebar.subheader("1️⃣ Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file (Indian_Kids_Screen_Time.csv)",
        type=['csv'],
        help="Upload the Indian Kids Screen Time dataset"
    )
    
    # Load data
    df = load_data(uploaded_file)
    
    if df is None or df.empty:
        st.error("❌ Unable to load data. Please upload a valid CSV file.")
        st.info("📋 Your CSV should have columns: Age, Gender, Avg_Daily_Screen_Time_hr, Primary_Device, Urban_or_Rural")
        st.stop()
    
    # Display data info
    with st.expander("📋 Dataset Overview", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Age Range", f"{int(df['Age'].min())}-{int(df['Age'].max())}")
        with col4:
            st.metric("Device Types", df['Device_Type'].nunique())
        
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show column names for reference
        st.write("**Available Columns:**", ", ".join(df.columns.tolist()))
    
    # ==================== SIDEBAR FILTERS ====================
    st.sidebar.subheader("2️⃣ Apply Filters")
    
    # Age Group Filter
    age_groups = ['All'] + sorted(df['Age_Group'].dropna().unique().tolist())
    selected_age = st.sidebar.selectbox("Select Age Group", age_groups)
    
    # Gender Filter
    genders = ['All'] + sorted(df['Gender'].unique().tolist())
    selected_gender = st.sidebar.selectbox("Select Gender", genders)
    
    # Device Type Filter
    devices = ['All'] + sorted(df['Device_Type'].unique().tolist())
    selected_device = st.sidebar.selectbox("Select Device Type", devices)
    
    # Location Filter
    locations = ['All'] + sorted(df['Location'].unique().tolist())
    selected_location = st.sidebar.selectbox("Select Location (Urban/Rural)", locations)
    
    # Health Impacts Filter (if available)
    if 'Health_Impacts' in df.columns:
        health_impacts = ['All'] + sorted(df['Health_Impacts'].unique().tolist())
        selected_health = st.sidebar.selectbox("Select Health Impact", health_impacts)
    else:
        selected_health = 'All'
    
    # Exceeded Limit Filter (if available)
    if 'Exceeded_Recommended_Limit' in df.columns:
        limit_options = ['All', 'Yes', 'No']
        selected_limit = st.sidebar.selectbox("Exceeded Recommended Limit?", limit_options)
    else:
        selected_limit = 'All'
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_age != 'All':
        filtered_df = filtered_df[filtered_df['Age_Group'] == selected_age]
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    if selected_device != 'All':
        filtered_df = filtered_df[filtered_df['Device_Type'] == selected_device]
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['Location'] == selected_location]
    if selected_health != 'All' and 'Health_Impacts' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Health_Impacts'] == selected_health]
    if selected_limit != 'All' and 'Exceeded_Recommended_Limit' in filtered_df.columns:
        if selected_limit == 'Yes':
            filtered_df = filtered_df[filtered_df['Exceeded_Recommended_Limit'] == True]
        else:
            filtered_df = filtered_df[filtered_df['Exceeded_Recommended_Limit'] == False]
    
    # Display filtered record count
    st.sidebar.success(f"📊 Filtered Records: {len(filtered_df)}")
    
    # ==================== KEY METRICS ====================
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_screen_time = filtered_df['Screen_Time_Hours'].mean()
        st.metric("Average Screen Time", f"{avg_screen_time:.2f} hrs")
    
    with col2:
        max_screen_time = filtered_df['Screen_Time_Hours'].max()
        st.metric("Maximum Screen Time", f"{max_screen_time:.2f} hrs")
    
    with col3:
        min_screen_time = filtered_df['Screen_Time_Hours'].min()
        st.metric("Minimum Screen Time", f"{min_screen_time:.2f} hrs")
    
    with col4:
        median_screen_time = filtered_df['Screen_Time_Hours'].median()
        st.metric("Median Screen Time", f"{median_screen_time:.2f} hrs")
    
    # Additional metrics if columns exist
    if 'Exceeded_Recommended_Limit' in filtered_df.columns:
        col5, col6 = st.columns(2)
        with col5:
            exceeded_pct = (filtered_df['Exceeded_Recommended_Limit'].sum() / len(filtered_df)) * 100
            st.metric("Exceeded Recommended Limit", f"{exceeded_pct:.1f}%")
        with col6:
            if 'Educational_to_Recreational_Ratio' in filtered_df.columns:
                avg_ratio = filtered_df['Educational_to_Recreational_Ratio'].mean()
                st.metric("Avg Education/Recreation Ratio", f"{avg_ratio:.2f}")
    
    st.markdown("---")
    
    # ==================== VISUALIZATIONS ====================
    st.subheader("📊 Interactive Visualizations")
    
    # Row 1: Age Group and Gender Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualization 1: Average Screen Time by Age Group
        st.markdown("### 🎂 Screen Time by Age Group")
        age_data = filtered_df.groupby('Age_Group')['Screen_Time_Hours'].mean().reset_index()
        age_data = age_data.sort_values('Screen_Time_Hours', ascending=False)
        
        fig1 = px.bar(
            age_data,
            x='Age_Group',
            y='Screen_Time_Hours',
            title='Average Screen Time by Age Group',
            labels={'Screen_Time_Hours': 'Average Hours', 'Age_Group': 'Age Group'},
            color='Screen_Time_Hours',
            color_continuous_scale='Blues',
            text='Screen_Time_Hours'
        )
        fig1.update_traces(texttemplate='%{text:.2f} hrs', textposition='outside')
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Visualization 2: Screen Time by Gender
        st.markdown("### 👥 Screen Time by Gender")
        gender_data = filtered_df.groupby('Gender')['Screen_Time_Hours'].mean().reset_index()
        
        fig2 = px.bar(
            gender_data,
            x='Gender',
            y='Screen_Time_Hours',
            title='Average Screen Time by Gender',
            labels={'Screen_Time_Hours': 'Average Hours', 'Gender': 'Gender'},
            color='Gender',
            color_discrete_sequence=['#ff7f0e', '#2ca02c'],
            text='Screen_Time_Hours'
        )
        fig2.update_traces(texttemplate='%{text:.2f} hrs', textposition='outside')
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Row 2: Device Distribution and Screen Time Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualization 3: Device Usage Distribution (Pie Chart)
        st.markdown("### 📱 Device Usage Distribution")
        device_data = filtered_df['Device_Type'].value_counts().reset_index()
        device_data.columns = ['Device_Type', 'Count']
        
        fig3 = px.pie(
            device_data,
            values='Count',
            names='Device_Type',
            title='Distribution of Device Types',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig3.update_traces(textposition='inside', textinfo='percent+label')
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Visualization 4: Screen Time Distribution (Histogram)
        st.markdown("### 📊 Screen Time Distribution")
        fig4 = px.histogram(
            filtered_df,
            x='Screen_Time_Hours',
            nbins=30,
            title='Distribution of Screen Time (Hours)',
            labels={'Screen_Time_Hours': 'Screen Time (Hours)', 'count': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        fig4.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Row 3: Health Impacts and Location Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualization 5: Health Impacts Distribution
        if 'Health_Impacts' in filtered_df.columns:
            st.markdown("### 🏥 Health Impacts Distribution")
            health_data = filtered_df['Health_Impacts'].value_counts().reset_index()
            health_data.columns = ['Health_Impacts', 'Count']
            health_data = health_data.sort_values('Count', ascending=True)
            
            fig5 = px.bar(
                health_data,
                y='Health_Impacts',
                x='Count',
                orientation='h',
                title='Frequency of Health Impacts',
                labels={'Count': 'Number of Cases', 'Health_Impacts': 'Health Impact'},
                color='Count',
                color_continuous_scale='Reds',
                text='Count'
            )
            fig5.update_traces(texttemplate='%{text}', textposition='outside')
            fig5.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        # Visualization 6: Urban vs Rural Comparison
        st.markdown("### 🏙️ Urban vs Rural Screen Time")
        location_data = filtered_df.groupby('Location')['Screen_Time_Hours'].mean().reset_index()
        
        fig6 = px.bar(
            location_data,
            x='Location',
            y='Screen_Time_Hours',
            title='Screen Time: Urban vs Rural',
            labels={'Screen_Time_Hours': 'Average Hours', 'Location': 'Location'},
            color='Location',
            color_discrete_sequence=['#d62728', '#9467bd'],
            text='Screen_Time_Hours'
        )
        fig6.update_traces(texttemplate='%{text:.2f} hrs', textposition='outside')
        fig6.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)
    
    # Row 4: Additional Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualization 7: Exceeded Limit by Age Group
        if 'Exceeded_Recommended_Limit' in filtered_df.columns:
            st.markdown("### ⚠️ Exceeded Limit by Age Group")
            limit_data = filtered_df.groupby('Age_Group')['Exceeded_Recommended_Limit'].apply(
                lambda x: (x.sum() / len(x)) * 100
            ).reset_index()
            limit_data.columns = ['Age_Group', 'Percentage']
            
            fig7 = px.bar(
                limit_data,
                x='Age_Group',
                y='Percentage',
                title='% Exceeding Recommended Screen Time Limit',
                labels={'Percentage': 'Percentage (%)', 'Age_Group': 'Age Group'},
                color='Percentage',
                color_continuous_scale='OrRd',
                text='Percentage'
            )
            fig7.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig7.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig7, use_container_width=True)
    
    with col2:
        # Visualization 8: Educational vs Recreational Ratio
        if 'Educational_to_Recreational_Ratio' in filtered_df.columns:
            st.markdown("### 📚 Education/Recreation Ratio by Age")
            ratio_data = filtered_df.groupby('Age_Group')['Educational_to_Recreational_Ratio'].mean().reset_index()
            
            fig8 = px.line(
                ratio_data,
                x='Age_Group',
                y='Educational_to_Recreational_Ratio',
                title='Average Education/Recreation Ratio by Age Group',
                labels={'Educational_to_Recreational_Ratio': 'Ratio', 'Age_Group': 'Age Group'},
                markers=True,
                color_discrete_sequence=['#17becf']
            )
            fig8.update_traces(line=dict(width=3), marker=dict(size=10))
            fig8.update_layout(height=400)
            st.plotly_chart(fig8, use_container_width=True)
    
    # Additional Visualization: Heatmap
    st.markdown("### 🔥 Screen Time Heatmap: Age Group vs Device Type")
    heatmap_data = filtered_df.pivot_table(
        values='Screen_Time_Hours',
        index='Age_Group',
        columns='Device_Type',
        aggfunc='mean'
    )
    
    fig9 = px.imshow(
        heatmap_data,
        title='Average Screen Time: Age Group vs Device Type',
        labels=dict(x="Device Type", y="Age Group", color="Avg Hours"),
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    fig9.update_layout(height=400)
    st.plotly_chart(fig9, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== KEY INSIGHTS ====================
    st.subheader("💡 Key Insights & Findings")
    
    # Calculate insights
    insights = generate_insights(df, filtered_df)
    
    # Display insights in styled boxes
    for i, insight in enumerate(insights, 1):
        st.markdown(f"""
            <div class="insight-box">
                <strong>Insight {i}:</strong> {insight}
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== FOOTER ====================
    st.markdown("""
        <div style='text-align: center; padding: 20px; color: #666;'>
            <p><strong>Indian Kids Screen Time Analysis Dashboard</strong></p>
            <p>Data Visualization Lab Assignment | 2025</p>
            <p>Built with Python, Streamlit & Plotly</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== INSIGHTS GENERATOR ====================
def generate_insights(df, filtered_df):
    """
    Generate key insights from the data analysis
    """
    insights = []
    
    # Insight 1: Highest screen time age group
    age_avg = df.groupby('Age_Group')['Screen_Time_Hours'].mean()
    highest_age = age_avg.idxmax()
    highest_value = age_avg.max()
    insights.append(
        f"<strong>Age Group Analysis:</strong> The {highest_age} age group has the highest average screen time "
        f"at {highest_value:.2f} hours per day, indicating increased digital engagement among older children."
    )
    
    # Insight 2: Most used device
    most_used_device = df['Device_Type'].value_counts().idxmax()
    device_percentage = (df['Device_Type'].value_counts().max() / len(df)) * 100
    insights.append(
        f"<strong>Device Preference:</strong> {most_used_device} is the most commonly used device, "
        f"accounting for {device_percentage:.1f}% of all usage, reflecting the ubiquity of mobile technology."
    )
    
    # Insight 3: Gender comparison
    gender_avg = df.groupby('Gender')['Screen_Time_Hours'].mean()
    if len(gender_avg) >= 2:
        gender_diff = abs(gender_avg.iloc[0] - gender_avg.iloc[1])
        if gender_diff > 0.5:
            higher_gender = gender_avg.idxmax()
            insights.append(
                f"<strong>Gender Differences:</strong> {higher_gender} students show {gender_diff:.2f} hours more "
                f"average screen time compared to the other gender, suggesting different usage patterns."
            )
        else:
            insights.append(
                f"<strong>Gender Parity:</strong> Screen time is relatively balanced between genders "
                f"with only {gender_diff:.2f} hours difference, indicating similar digital habits."
            )
    
    # Insight 4: Exceeded Recommended Limit
    if 'Exceeded_Recommended_Limit' in df.columns:
        exceeded_pct = (df['Exceeded_Recommended_Limit'].sum() / len(df)) * 100
        insights.append(
            f"<strong>Recommended Limit:</strong> {exceeded_pct:.1f}% of children exceed the recommended daily "
            f"screen time limit, highlighting a widespread concern for digital wellness among Indian youth."
        )
    
    # Insight 5: Urban vs Rural
    location_avg = df.groupby('Location')['Screen_Time_Hours'].mean()
    if len(location_avg) >= 2:
        urban_time = location_avg.get('Urban', 0)
        rural_time = location_avg.get('Rural', 0)
        diff = abs(urban_time - rural_time)
        higher_loc = location_avg.idxmax()
        insights.append(
            f"<strong>Geographic Divide:</strong> {higher_loc} areas show {diff:.2f} hours more screen time, "
            f"potentially reflecting differences in digital infrastructure and lifestyle patterns."
        )
    
    # Insight 6: Health Impacts
    if 'Health_Impacts' in df.columns:
        most_common_health = df['Health_Impacts'].value_counts().idxmax()
        health_pct = (df['Health_Impacts'].value_counts().max() / len(df)) * 100
        insights.append(
            f"<strong>Health Concerns:</strong> {most_common_health} is the most reported health impact ({health_pct:.1f}%), "
            f"emphasizing the need for awareness about healthy screen usage habits."
        )
    
    # Insight 7: Educational vs Recreational
    if 'Educational_to_Recreational_Ratio' in df.columns:
        avg_ratio = df['Educational_to_Recreational_Ratio'].mean()
        if avg_ratio < 0.3:
            insights.append(
                f"<strong>Usage Purpose:</strong> The average education-to-recreation ratio is {avg_ratio:.2f}, "
                f"indicating that recreational use dominates screen time, suggesting opportunities for more educational content."
            )
        elif avg_ratio > 0.6:
            insights.append(
                f"<strong>Usage Purpose:</strong> The average education-to-recreation ratio is {avg_ratio:.2f}, "
                f"showing a healthy balance towards educational content consumption."
            )
        else:
            insights.append(
                f"<strong>Usage Purpose:</strong> The average education-to-recreation ratio is {avg_ratio:.2f}, "
                f"indicating a moderate balance between educational and recreational screen time."
            )
    
    return insights

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()