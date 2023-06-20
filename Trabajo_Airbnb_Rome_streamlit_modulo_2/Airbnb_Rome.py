#Librerias
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import os
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# mapas interactivos
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
#import leafmap.kepler as leafmap
from branca.colormap import LinearColormap
from streamlit_folium import folium_static
from streamlit_folium import st_folium
import pydeck as pdk

#to make the plotly graphs
import plotly.graph_objs as go
import chart_studio.plotly as py
import plotly.figure_factory as ff
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import plotly.express as px
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

#text mining
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer



#Configuración de página, puede ser "centered" o "wide"
st.set_page_config(page_title="Bienvenidos a Roma", 
                   layout="wide",
                   page_icon=":it:",
                   initial_sidebar_state= 'auto')
st.set_option('deprecation.showPyplotGlobalUse', False)



# C:\Users\Samplerepo\Bootcamp_analytics\Trabajo_Airbnb_Rome_streamlit_modulo_2 
# streamlit run 1_Airbnb_Rome.py

#Creación de columnas y logotipo
st.image("Img\Roma estrecho.jpg", use_column_width=True)

# Título de la página
st.title("1. Bienvenidos a Roma de la mano de Airbnb")

# Descripción 
st.markdown("""Airbnb es una compañía que ofrece una plataforma digital dedicada a la oferta de alojamientos a particulares 
mediante la cual los anfitriones pueden publicitar y arrendar sus propiedades. Tanto anfitriones como huéspedes pueden valorarse 
mutuamente como referencia para futuros usuarios.Desde la página Insideairbnb.com he descargado los archivos de la ciudad de Roma
 a fecha de 15 de Marzo de 2023 y a través de ellos conocermos un poco mejor 'La Ciudad Eterna'.
""" )

#Librerias utilizadas
st.header("1.1. Librerias y utilidades")
st.code ("""# librerias básicas
import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# mapas interactivos
import folium
from folium.plugins import FastMarkerCluster
import geopandas as gpd
from branca.colormap import LinearColormap

# gráficas
import matplotlib
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import chart_studio.plotly as py
from plotly.offline import iplot, init_notebook_mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

# minado de texto
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer""")

st.header("1.2. Procesamiento de datos")
st.markdown("""
En este apartado se realizará la carga de datos, la visualización de valores perdidos y la extracción de características. 
El conjunto de datos original de la pagina Insideairbnb.com consta de 7 archivos, aunque no vamos a trabajar con todos ellos.
""" )


# ocultar y expandir código CSV
st.subheader('1.2.1. Cargando datos')
st.markdown("""
 A continuación se muestran dos dataframe sobre los que se va a trabajar principalmente y una descripción de ellos .""" )
def main():
    with st.expander("Mostrar código"):
        st.code ("""listings = pd.read_csv(r"Data/listings.csv")
reviews = pd.read_csv(r"Data/reviews.csv", parse_dates=['date'])
listings_details = pd.read_csv(r"Data/listings.csv.gz", index_col= "id")
calendar = pd.read_csv(r"Data/calendar.csv.gz",  parse_dates=['date'], index_col=['listing_id'])
reviews_details = pd.read_csv(r"Data/reviews.csv.gz", parse_dates=['date'])
Rome_monuments_locations = pd.read_csv(r"Data/Rome_monuments_locations.csv")
""")
if __name__ == '__main__':
    main()

# dataframes
st.markdown("Archivo listings.csv")
listings = pd.read_csv(r"Data/listings.csv")
listings
st.text("""<class 'pandas.core.frame.DataFrame'>
Int64Index: 24924 entries, 256695 to 843370708957028027
Data columns (total 17 columns):
 #   Column                          Non-Null Count  Dtype  
---  ------                          --------------  -----  
 0   name                            24920 non-null  object 
 1   host_id                         24924 non-null  int64  
 2   host_name                       24916 non-null  object 
 3   neighbourhood_group             0 non-null      float64
 4   neighbourhood                   24924 non-null  object 
 5   latitude                        24924 non-null  float64
 6   longitude                       24924 non-null  float64
 7   room_type                       24924 non-null  object 
 8   price                           24924 non-null  int64  
 9   minimum_nights                  24924 non-null  int64  
 10  number_of_reviews               24924 non-null  int64  
 11  last_review                     21152 non-null  object 
 12  reviews_per_month               21152 non-null  float64
 13  calculated_host_listings_count  24924 non-null  int64  
 14  availability_365                24924 non-null  int64  
 15  number_of_reviews_ltm           24924 non-null  int64  
 16  license                         4455 non-null   object 
dtypes: float64(4), int64(7), object(6)
memory usage: 3.4+ MB""")

st.markdown("Archivo listings_details.csv")
listings_details = pd.read_csv(r"Data/listings_details.csv", index_col= "id")
listings_details
st.text("""<class 'pandas.core.frame.DataFrame'>
Int64Index: 24924 entries, 256695 to 846391462079596101
Data columns (total 74 columns):
 #   Column                                        Non-Null Count  Dtype  
---  ------                                        --------------  -----  
 0   listing_url                                   24924 non-null  object 
 1   scrape_id                                     24924 non-null  int64  
 2   last_scraped                                  24924 non-null  object 
 3   source                                        24924 non-null  object 
 4   name                                          24920 non-null  object 
 5   description                                   24465 non-null  object 
 6   neighborhood_overview                         15266 non-null  object 
 7   picture_url                                   24924 non-null  object 
 8   host_id                                       24924 non-null  int64  
 9   host_url                                      24924 non-null  object 
 10  host_name                                     24916 non-null  object 
 11  host_since                                    24916 non-null  object 
 12  host_location                                 19419 non-null  object 
 13  host_about                                    13536 non-null  object 
 14  host_response_time                            21101 non-null  object 
 15  host_response_rate                            21101 non-null  object 
 16  host_acceptance_rate                          22464 non-null  object 
 17  host_is_superhost                             24901 non-null  object 
 18  host_thumbnail_url                            24916 non-null  object 
 19  host_picture_url                              24916 non-null  object 
...
 72  calculated_host_listings_count_shared_rooms   24924 non-null  int64  
 73  reviews_per_month                             21152 non-null  float64
dtypes: float64(23), int64(16), object(35)
memory usage: 14.3+ MB""")



#preparando el df
st.subheader('1.2.2. Preparando el dataframe principal de trabajo')
st.markdown("""El archivo listados_detalles contiene un total de 74 variables. 
No vamos a usar todas ellas, sino que juntaremos una serie de variables de los archivos listings y listings details que utilizaremos para nuestro analisis.""")
def main():
    with st.expander("Mostrar código"):
        st.code ("""target_columns = ['description','price','bedrooms','beds','amenities','property_type', 'accommodates', 'first_review', 'review_scores_value', 'review_scores_cleanliness', 'review_scores_location', 'review_scores_accuracy', 'review_scores_communication', 'review_scores_checkin', 'review_scores_rating', 'maximum_nights', 'listing_url', 'host_is_superhost', 'host_about', 'host_response_time', 'host_response_rate']
listings = pd.merge(listings, listings_details[target_columns], on='id', how='left')
listings.info()""")
if __name__ == '__main__':
    main()

# df
target_columns = ['description','price','bedrooms','beds','amenities','property_type', 'accommodates', 'first_review', 'review_scores_value', 'review_scores_cleanliness', 'review_scores_location', 'review_scores_accuracy', 'review_scores_communication', 'review_scores_checkin', 'review_scores_rating', 'maximum_nights', 'listing_url', 'host_is_superhost', 'host_about', 'host_response_time', 'host_response_rate']
listings = pd.merge(listings, listings_details[target_columns], on='id', how='left')

#buscando valores nulos 
st.subheader('1.2.3.  Buscando valores nulos en el nuevo dataframe listings')
st.markdown(""" Se puede observar que hay alguna variables como neighbourhood_group y license con un alto porcentaje faltante. 
Posteriormente pasaremos a eliminarlas
""")
def main():
    with st.expander("Mostrar código"):
        st.code ("""listings_nan_percentage = (listings.isnull().mean() * 100).sort_values(ascending=False)""")
if __name__ == '__main__':
    main()

st.text("""
neighbourhood_group               100.000000
license                            82.125662
last_review                        15.134007
reviews_per_month                  15.134007
host_name                           0.032098
name                                0.016049
longitude                           0.000000
room_type                           0.000000
host_id                             0.000000
minimum_nights                      0.000000
number_of_reviews                   0.000000
latitude                            0.000000
neighbourhood                       0.000000
calculated_host_listings_count      0.000000
availability_365                    0.000000
number_of_reviews_ltm               0.000000
price                               0.000000
dtype: float64
""")

#calculo el porcentaje de valores nulos en cada columna
listings_nan_percentage = (listings.isnull().mean() * 100).sort_values(ascending=False).round(1)
#gráfico de barras utilizando Plotly Express 
fig = px.bar(listings_nan_percentage, x=listings_nan_percentage.index, y=listings_nan_percentage, width=800)
fig.update_layout(
    xaxis_title="Variables",
    yaxis_title="Percentage (%) of nan")
st.plotly_chart(fig, use_container_width=True)

st.markdown(""" Se puede observar que hay alguna variables como 'neighbourhood_group' y 'license' con un alto porcentaje faltante.
 Posteriormente pasaremos a eliminarlas
 """)

#trabajando la columna price
st.subheader("1.2.4.  Valores columna 'price_x'")
st.markdown("""Parece que hay una gran disparidad en los precios de las viviendas, habitaciones, etc. 
Por ello pasamos a primero realizar un analisis donde se puede ver que el alquiler mas baratos 
es de 0 \$ y el mas caro 100180 \$, con una std de un 1841%
""")
def main():
    with st.expander("Mostrar código"):
        st.code ("""listings['price_x'].describe()""")
if __name__ == '__main__':
    main()
st.text("""
count     24924.000000
mean        214.729819
std        1841.303238
min           0.000000
25%          72.000000
50%         105.000000
75%         161.000000
max      100180.000000
Name: price_x, dtype: float64
""")

#gráfico de caja utilizando Plotly Express
fig = px.box(listings, y='price_x', notched=True, width=800)
fig.update_layout(
    title = 'Price variable with outliers',
    xaxis_title="Variables",
    yaxis_title="Precio ($)")
st.plotly_chart(fig)

#calculamos la mediana de los precios agrupados por barrios
median_prices = listings.groupby('neighbourhood')['price_x'].median().sort_values(ascending=False)
#los valores 0 los cambiamos por la mediana de los precios de la columna price_x
listings['price_x']=listings['price_x'].replace(0,listings['price_x'].median())
# Calcular el rango intercuartilico para identificar los valores atípicos
Q1 = listings['price_x'].quantile(0.25)
Q3 = listings['price_x'].quantile(0.75)
IQR = Q3 - Q1
# Identificar los valores atípicos
outliers = (listings['price_x'] < Q1 - 1.5 * IQR) | (listings['price_x'] > Q3 + 1.5 * IQR)
# Crear una nueva columna llamada 'price_clean' y copiar los valores originales
listings['price_clean'] = listings['price_x']
# Sustituir los valores atípicos por la mediana correspondiente en la nueva columna
for index, row in listings[outliers].iterrows():
    neighborhood = row['neighbourhood']
    median_price = median_prices[neighborhood]#ya esta calculado mas arriba
    listings.at[index, 'price_clean'] = median_price

st.markdown("""Para intentar solucionar esto calculamos el IQR de los precios y sustituimos los valores atípicos por la mediana 
de los precios agrupados por barrios y cremos una nueva columna llamada price_clean
""")
def main():
    with st.expander("Mostrar código"):
        st.code ("""# calculamos la mediana de los precios agrupados por barrios
median_prices = listings.groupby('neighbourhood')['price_x'].median().sort_values(ascending=False)
# los valores 0 los cambiamos por la mediana de los precios de la columna price_x
listings['price_x']=listings['price_x'].replace(0,listings['price_x'].median())
# calculamos el rango intercuartilico para identificar los valores atípicos
Q1 = listings['price_x'].quantile(0.25)
Q3 = listings['price_x'].quantile(0.75)
IQR = Q3 - Q1
# identificmos los valores atípicos
outliers = (listings['price_x'] < Q1 - 1.5 * IQR) | (listings['price_x'] > Q3 + 1.5 * IQR)
# creamos una nueva columna llamada 'price_clean' y copiamos los valores originales
listings['price_clean'] = listings['price_x']
# sutituimos  los valores atípicos por la mediana correspondiente en la nueva columna
for index, row in listings[outliers].iterrows():
    neighborhood = row['neighbourhood']
    median_price = median_prices[neighborhood]#ya esta calculado mas arriba
    listings.at[index, 'price_clean'] = median_price
""")
if __name__ == '__main__':
    main()

#gráfico de caja utilizando Plotly Express
fig2 = px.box(listings, y='price_clean', notched=True, width=800)
fig2.update_layout(
    title = 'New price variable without outliers',
    xaxis_title="Variables",
    yaxis_title="Precio ($)")
st.plotly_chart(fig2)

st.markdown("""Una vez aplicados los cambios parece que los precios de los alquileres de las viviendas y 
habitaciones tienen unos valores con algo más de sentido.
""")
def main():
    with st.expander("Mostrar código"):
        st.code ("""listings['price_clean'].describe()""")
if __name__ == '__main__':
    main()
st.text("""
count    24924.000000
mean       112.359011
std         55.523917
min          8.000000
25%         71.000000
50%        101.000000
75%        138.000000
max        294.000000
Name: price_clean, dtype: float64
""")

# trabajando el resto de variables
st.subheader('1.2.4.  Resto de variables')
st.markdown("""Con el resto de variables rellenaremos los valores faltantes con las medianas, 
medias o modas según sean necesarias.""")

# solo asigando a una variable, no cambio el df (inplace=True)
listings['host_response_rate'] = pd.to_numeric(listings['host_response_rate'].str.strip('%'))
listings['review_scores_cleanliness'] = listings['review_scores_cleanliness'].fillna(listings['review_scores_cleanliness'].mean().round(2))
listings['review_scores_location'] = listings['review_scores_location'].fillna(listings['review_scores_location'].mean().round(2))
listings['review_scores_accuracy'] = listings['review_scores_accuracy'].fillna(listings['review_scores_accuracy'].mean().round(2))
listings['review_scores_communication'] = listings['review_scores_communication'].fillna(listings['review_scores_communication'].mean().round(2))
listings['review_scores_checkin'] = listings['review_scores_checkin'].fillna(listings['review_scores_checkin'].mean().round(2))
listings['review_scores_rating'] = listings['review_scores_rating'].fillna(listings['review_scores_rating'].mean().round(2))
listings['beds'] = listings['beds'].fillna(listings['beds'].median())
listings['bedrooms'] = listings['bedrooms'].fillna(listings['bedrooms'].median())
listings['host_response_rate'] = listings['host_response_rate'].fillna(listings['host_response_rate'].mean().round(2))

def main():
    with st.expander("Mostrar código"):
        st.code ("""listings['host_response_rate'] = pd.to_numeric(listings['host_response_rate'].str.strip('%'))
listings['review_scores_value'] = listings['review_scores_value'].fillna(listings['review_scores_value'].mean().round(2))
listings['review_scores_cleanliness'] = listings['review_scores_cleanliness'].fillna(listings['review_scores_cleanliness'].mean().round(2))
listings['review_scores_location'] = listings['review_scores_location'].fillna(listings['review_scores_location'].mean().round(2))
listings['review_scores_accuracy'] = listings['review_scores_accuracy'].fillna(listings['review_scores_accuracy'].mean().round(2))
listings['review_scores_communication'] = listings['review_scores_communication'].fillna(listings['review_scores_communication'].mean().round(2))
listings['review_scores_checkin'] = listings['review_scores_checkin'].fillna(listings['review_scores_checkin'].mean().round(2))
listings['review_scores_rating'] = listings['review_scores_rating'].fillna(listings['review_scores_rating'].mean().round(2))
listings['beds'] = listings['beds'].fillna(listings['beds'].median())
listings['bedrooms'] = listings['bedrooms'].fillna(listings['bedrooms'].median())
listings['host_response_rate'] = listings['host_response_rate'].fillna(listings['host_response_rate'].mean().round(2))""")
if __name__ == '__main__':
    main()


st.text("""
name                                  4
host_id                               0
host_name                             8
neighbourhood_group               24924
neighbourhood                         0
latitude                              0
longitude                             0
room_type                             0
price_x                               0
minimum_nights                        0
number_of_reviews                     0
last_review                        3772
reviews_per_month                  3772
calculated_host_listings_count        0
availability_365                      0
number_of_reviews_ltm                 0
license                           20469
description                         459
price_y                               0
bedrooms                              0
beds                                  0
amenities                             0
property_type                         0
accommodates                          0
first_review                       3772
...
host_about                        11388
host_response_time                 3823
host_response_rate                    0
price_clean                           0
dtype: int64
""")


st.subheader("Correlacion entre variables")
st.markdown("""Realizamos un heatmap para ver si hay fuertes correlaciones entre algunas de nuestras variables y asi poder decidir si eliminamos alguna
""")
def main():
    with st.expander("Mostrar código"):
        st.code ("""
corr = listings.corr(method = 'pearson').sort_values(by = 'price_clean', axis = 0, ascending = False).sort_values(by = 'price_clean', axis = 1, ascending = False)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15,15))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr.iloc[0:37,0:37], mask=mask[0:37,0:37], cmap='PuOr', vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True, fmt=".2f")
st.pyplot(f)""")
if __name__ == '__main__':
    main()

st.markdown("""
En nuestro caso hay fuertes relaciones entre algunas variables como por ejemplo:
* beds - accommodates.
* review_score_values - review_score_rating.
* review_per_mouth - number_of_reviews_ltm.
* review_score_accuracy - review_score_value. """)

corr = listings.corr(method = 'pearson').sort_values(by = 'price_clean', axis = 0, ascending = False).sort_values(by = 'price_clean', axis = 1, ascending = False)
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15,15))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr.iloc[0:37,0:37], mask=mask[0:37,0:37], cmap='PuOr', vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot = True, fmt=".2f")
st.pyplot(f)


st.markdown("""Como comentabamos anteriormente, hay algunas columnas repetidas y en otras hay gran 
cantidad de valores faltantes que procederemos a eliminar y las columnas restantes las ordenaremos de una manera mas lógica.""")

def main():
    with st.expander("Mostrar código"):
        st.code ("""listings['price_x'].describe()

new_order = ['host_id', 'host_name','host_is_superhost', 'host_about', 'name', 'description', 'room_type', 'neighbourhood', 'latitude',
       'longitude', 'price_clean', 'minimum_nights', 'maximum_nights',
       'number_of_reviews', 'reviews_per_month',
       'calculated_host_listings_count', 'availability_365',
       'number_of_reviews_ltm', 'property_type',
       'accommodates','bedrooms','beds','amenities', 'first_review', 'last_review', 'review_scores_value',
       'review_scores_cleanliness', 'review_scores_location',
       'review_scores_accuracy', 'review_scores_communication',
       'review_scores_checkin', 'review_scores_rating', 
       'listing_url', 'host_response_time',
       'host_response_rate']

listings = listings.reindex(columns=new_order)""")
if __name__ == '__main__':
    main()

# eliminamos columnas
listings = listings.drop(columns=['neighbourhood_group','license'])

new_order = ['host_id', 'host_name','host_is_superhost', 'host_about', 'name', 'description', 'room_type', 'neighbourhood', 'latitude',
       'longitude','price_y','price_x', 'price_clean', 'minimum_nights', 'maximum_nights',
       'number_of_reviews',  'reviews_per_month',
       'calculated_host_listings_count', 'availability_365',
       'number_of_reviews_ltm', 'property_type',
       'accommodates','bedrooms','beds','amenities', 'first_review', 'last_review', 'review_scores_value',
       'review_scores_cleanliness', 'review_scores_location',
       'review_scores_accuracy', 'review_scores_communication',
       'review_scores_checkin', 'review_scores_rating', 
       'listing_url', 'host_response_time',
       'host_response_rate']

listings = listings.reindex(columns=new_order)

st.subheader("Finalmente obtenemos el dataframe con el que viajaremos a Roma")
st.markdown("Archivo listings_depurado.csv")
listings_depurado = pd.read_csv(r"Data/listings_depurado.csv")
listings_depurado

st.text("""<class 'pandas.core.frame.DataFrame'>
RangeIndex: 24924 entries, 0 to 24923
Data columns (total 35 columns):
 #   Column                          Non-Null Count  Dtype  
---  ------                          --------------  -----  
 0   host_id                         24924 non-null  int64  
 1   host_name                       24916 non-null  object 
 2   host_is_superhost               24901 non-null  object 
 3   host_about                      13536 non-null  object 
 4   name                            24920 non-null  object 
 5   description                     24465 non-null  object 
 6   room_type                       24924 non-null  object 
 7   neighbourhood                   24924 non-null  object 
 8   latitude                        24924 non-null  float64
 9   longitude                       24924 non-null  float64
 10  price_clean                     24924 non-null  int64  
 11  minimum_nights                  24924 non-null  int64  
 12  maximum_nights                  24924 non-null  int64  
 13  number_of_reviews               24924 non-null  int64  
 14  reviews_per_month               21152 non-null  float64
 15  calculated_host_listings_count  24924 non-null  int64  
 16  availability_365                24924 non-null  int64  
 17  number_of_reviews_ltm           24924 non-null  int64  
 18  property_type                   24924 non-null  object 
 19  accommodates                    24924 non-null  int64  
...
 33  host_response_time              21101 non-null  object 
 34  host_response_rate              24924 non-null  float64
dtypes: float64(13), int64(9), object(13)
memory usage: 6.7+ MB""")


#page_bg_img = """
#<style>
#[data-testid="stAppViewContainer"] {
#background-image: url("https://es.123rf.com/photo_81692293_iconos-antiguos-de-color-roma-srt-para-dise%C3%B1o-web-y-m%C3%B3vil.html");
#background-size: cover;
#}

#[data-testid="stHeader"] {
#background-color: rgba(0,0,0,0);
#}

#<style>
#"""
#st.markdown(page_bg_img, unsafe_allow_html=True)