import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from babel.numbers import format_currency
sns.set(style='dark')
import numpy as np


# Load data
all_df = pd.read_csv("all_data.csv")

# Function to get top selling products
def get_top_selling_products(df):
    df_large_product_sales = df.groupby(by="product_name").agg({
        "payment_value": "sum",
    }).nlargest(10, "payment_value").round().reset_index()
    return df_large_product_sales

# Function to get least selling products
def get_least_selling_products(df):
    df_small_product_sales = df.groupby(by="product_name").agg({
        "payment_value": "sum",
    }).nsmallest(10, "payment_value").round().reset_index()
    return df_small_product_sales

def get_payment_type_info(df):
    df_payment_type = df.groupby(by="payment_type").agg({
        "customer_id": "count",
    }).reset_index()
    return df_payment_type

def rfm_analysis(df):
    # Convert date columns to datetime
    datetime_columns = ["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]
    for column in datetime_columns:
        df[column] = pd.to_datetime(df[column])
        df[column] = df[column].dt.date
        df[column] = pd.to_datetime(df[column])

    # Calculate recency (R)
    df_recency = df.groupby(by='customer_id', as_index=False)['order_approved_at'].max()
    df_recency.columns = ['CustomerId', 'LastPurchaseDate']
    recent_date = df_recency['LastPurchaseDate'].max()
    df_recency['Recency'] = df_recency['LastPurchaseDate'].apply(lambda x: (recent_date - x).days)

    # Calculate frequency (F)
    frequency_df = df.drop_duplicates().groupby(by=['customer_id'], as_index=False)['order_approved_at'].count()
    frequency_df.columns = ['CustomerId', 'Frequency']

    # Calculate monetary (M)
    monetary_df = df.groupby(by='customer_id', as_index=False)['payment_value'].sum()
    monetary_df.columns = ['CustomerId', 'Monetary']

    # Merge RFM data
    rf_df = df_recency.merge(frequency_df, on='CustomerId')
    rfm_df = rf_df.merge(monetary_df, on='CustomerId').drop(columns='LastPurchaseDate')

    # Rank RFM values
    rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending=False)
    rfm_df['F_rank'] = rfm_df['Frequency'].rank(ascending=True)
    rfm_df['M_rank'] = rfm_df['Monetary'].rank(ascending=True)

    # Normalizing the rank of the customers
    rfm_df['R_rank_norm'] = (rfm_df['R_rank']/rfm_df['R_rank'].max())*100
    rfm_df['F_rank_norm'] = (rfm_df['F_rank']/rfm_df['F_rank'].max())*100
    rfm_df['M_rank_norm'] = (rfm_df['F_rank']/rfm_df['M_rank'].max())*100

    # Drop unnecessary columns
    rfm_df.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

    # Calculate RFM Score
    rfm_df['RFM_Score'] = 0.15*rfm_df['R_rank_norm'] + 0.28*rfm_df['F_rank_norm'] + 0.57*rfm_df['M_rank_norm']
    rfm_df['RFM_Score'] *= 0.05
    rfm_df = rfm_df.round(2)

    # Assign customer segments
    rfm_df["Customer_segment"] = np.where(rfm_df['RFM_Score'] > 4.5, "Top Customers",
                                          (np.where(rfm_df['RFM_Score'] > 4,
                                                     "High value Customer",
                                                     (np.where(rfm_df['RFM_Score'] > 3,
                                                                "Medium Value Customer",
                                                                np.where(rfm_df['RFM_Score'] > 1.6,
                                                                         'Low Value Customers', 'Lost Customers'))))))
    return rfm_df


# Get top and least selling products
df_large_product_sales = get_top_selling_products(all_df)
df_small_product_sales = get_least_selling_products(all_df)
df_payment_type = get_payment_type_info(all_df)
rfm_df = rfm_analysis(all_df)


# Sidebar
with st.sidebar:
    st.title("Rizka Dwi Arzita")
    st.image("logo_dash.jpg", width=100)
    st.title("DASHBOARD")
    selected = st.radio("Navigation", ["About Projects", "Dataset Overview", "Payment Types",  "RFM Analysis"])

# Main content
st.title('Dashboard E-commerce')

if selected == "About Projects":
    st.image("logo_dash.jpg", width=500)
    st.write("This is a simple dashboard created using Streamlit.")
    st.subheader("Business Questions & Analysis Plan")
    st.markdown("### Business Questions:")
    st.markdown("1. **Top Selling Products:** Mengidentifikasi produk-produk dengan tingkat penjualan tertinggi dan terendah.")
    st.markdown("2. **Payment Types:** Menentukan jenis pembayaran yang paling banyak digunakan oleh pelanggan.")
    st.markdown("3. **RFM Analysis:** Menganalisis tingkat Recency, Frequency, dan Monetary dari pelanggan yang ada.")
    
    st.markdown("### Analysis Plan:")
    st.markdown("1. **Top Selling Products:** Analisis dilakukan dengan menghitung total penjualan untuk setiap produk, kemudian mengidentifikasi 10 produk dengan penjualan tertinggi dan terendah.")
    st.markdown("2. **Payment Types:** Menghitung jumlah penggunaan setiap jenis pembayaran oleh pelanggan.")
    st.markdown("3. **RFM Analysis:** Analisis RFM (Recency, Frequency, Monetary) digunakan untuk membagi pelanggan menjadi segmen berbeda berdasarkan perilaku transaksi mereka.")
elif selected == "Dataset Overview":
    st.subheader("Top Selling Products")
    fig1 = go.Figure()

    fig1.add_trace(go.Bar(
        x=df_large_product_sales['product_name'],
        y=df_large_product_sales['payment_value'],
        marker=dict(
            color='rgb(173, 216, 230)',  # Set color here
        ),
    ))

    for i, v in enumerate(df_large_product_sales['payment_value']):
        fig1.add_annotation(
            x=df_large_product_sales['product_name'][i],
            y=v+5,
            text=str(v),
            font=dict(
                size=10,
                color='black',  # Set text color here
            ),
            showarrow=False,
            textangle=90,
            align='center',
        )

    fig1.update_layout(
        title='Top 10 Best Selling Products 2016 - 2018',
        xaxis=dict(
            title='Product Name',
            tickangle=45,
            tickfont=dict(
                size=12,
            ),
            automargin=True,
        ),
        yaxis=dict(
            title='Total Sales',
            tickformat='plain',
            tickfont=dict(
                size=12,
            ),
        ),

        showlegend=False,
    )

    st.plotly_chart(fig1)

    st.subheader("Least Selling Products")
    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=df_small_product_sales['product_name'],
        y=df_small_product_sales['payment_value'],
        marker=dict(
            color='rgb(255, 204, 204)',  # Set color here
        ),
    ))

    for i, v in enumerate(df_small_product_sales['payment_value']):
        fig2.add_annotation(
            x=df_small_product_sales['product_name'][i],
            y=v+5,
            text=str(v),
            font=dict(
                size=10,
                color='black',  # Set text color here
            ),
            showarrow=False,
            textangle=90,
            align='center',
        )

    fig2.update_layout(
        title='Top 10 Least Selling Products 2016 - 2018',
        xaxis=dict(
            title='Product Name',
            tickangle=45,
            tickfont=dict(
                size=12,
            ),
            automargin=True,
        ),
        yaxis=dict(
            title='Total Sales',
            tickformat='plain',
            tickfont=dict(
                size=12,
            ),
        ),

        showlegend=False,
    )

    st.plotly_chart(fig2)

    with st.expander("Conclusion", expanded=False):

            # Konten expander
            st.write("Berdasarkan output yang telah diperoleh bahwa top 10 kategori produk yang memiliki tingkat penjualan paling tinggi diantaranya bed bath table, health beauty, computers accesories, furniture decor, watches gifts, sports leisure, housewares, garden tools, auto, dan cool stuff menunjukkan bahwa produk-produk dalam kategori tersebut memiliki permintaan yang tinggi di pasar. Namun, dapat dilihat juga kategori produk dengan tingkat penjualan paling rendah yakni security and service, fashion childrens clothes, cds dvds musicals, home comfort, flowers, art and craftmanship, la cuisine, fashion sport, diapers and hygiene, fashion female clothing.Jumlah penjualan tertinggi mencapai 1.725.466 barang dan jumlah penjualan terendah yaitu sebesar 325 barang. Hal ini menunjukkan variasi dalam preferensi konsumen dan potensi pasar untuk berbagai jenis produk.")
           
# Get payment type information
df_payment_type = get_payment_type_info(all_df)

if selected == "Payment Types":
    st.subheader("Number of Customers by Payment Type")

    # Plot the bar chart
    fig3 = go.Figure()

    fig3.add_trace(go.Bar(
        x=df_payment_type['payment_type'],
        y=df_payment_type['customer_id'],
        marker=dict(
            color='rgb(173, 216, 230)',  # Set color here
        ),
    ))

    for i, v in enumerate(df_payment_type['customer_id']):
        fig3.add_annotation(
            x=df_payment_type['payment_type'][i],
            y=v+5,
            text=str(v),
            font=dict(
                size=10,
                color='black',  # Set text color here
            ),
            showarrow=False,
            textangle=0,
            align='center',
        )

    fig3.update_layout(
        title='Number of Customers by Payment Type',
        xaxis=dict(
            title='Payment Type',
            tickangle=45,
            tickfont=dict(
                size=12,
            ),
            automargin=True,
        ),
        yaxis=dict(
            title='Number of Customers',
            tickformat='plain',
            tickfont=dict(
                size=12,
            ),
        ),

        showlegend=False,
    )

    st.plotly_chart(fig3)

    with st.expander("Conclusion", expanded=False):

            # Konten expander
            st.write("Berdasarkan visualisasi data yang telah diperoleh, dapat diketahui bahwa jenis pembayaran yang mendominasi customers adalah credit card.")
           
# Plot barplot
rfm_df = rfm_analysis(all_df)

if selected == "RFM Analysis":
    st.subheader("Recency, Frequency, and Monetary Analysis")

    segment_counts = rfm_df['Customer_segment'].value_counts()
    plt.figure(figsize=(15, 6))
    sns.barplot(x=segment_counts.values, y=segment_counts.index, palette='pastel')
    
    for i, v in enumerate(segment_counts.values):
      plt.text(v + 0.1, i, f'{(v / segment_counts.sum() * 100):.2f}%', color='black', va='center')
    
    plt.title('Customer Segmentation')
    plt.xlabel('Count')
    plt.ylabel('Segment')
    st.pyplot(plt)

    with st.expander("Conclusion", expanded=False):

            # Konten expander
            st.write("Mayoritas customers termasuk dalam kategori Low Value Customer yang menunjukkan bahwa mereka memiliki nilai RFM score yang rendah, yaitu di bawah 1.6 karena banyak pelanggan yang hanya melakukan satu kali transaksi dan tidak melakukan transaksi lagi selama periode tahun 2016 hingga 2018. Terdapat sebagian kecil dari seluruh pelanggan, yakni 3.52%, termasuk dalam kategori Top Customer sementara hanya 8.14% yang termasuk dalam kategori High Value Customer. Meskipun demikian, proporsi pelanggan dengan nilai RFM yang tinggi ini relatif kecil dibandingkan dengan pelanggan lainnya.")
           


st.caption('Copyright (C) Rizka Dwi Arzita 2024')
