import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import plotly.express as px
import plotly.graph_objects as go

# Streamlit app layout
st.title("Case 3: Flights")

# Uitleg over de app
st.write("""
    Groep 11+18: Leon Kourzanov, Tirej Sendi, Jason Shi, Luuk Terweijden, Koen van Hardeveld.
         
    Dit dashboard biedt een overzicht van de vluchtstatussen van verschillende luchthavens. 
    Je kunt gegevens bekijken op basis van tijdzones, trends in vluchtschema's analyseren 
    en de vluchtstatus van verschillende luchthavens visualiseren op een kaart.
""")

# Laad de luchthavengegevens van CSV
airports = pd.read_csv('schedule_airport.csv')
airports_cleaned = airports.dropna(subset=['Org/Des'])

st.subheader("Data Cleaning")
st.code("airports_cleaned = airports.dropna(subset=['Org/Des'])")

st.code("duplicates = airports_cleaned[airports_cleaned.duplicated()]")
duplicates = airports_cleaned[airports_cleaned.duplicated()]

# boxplot outliers
# Select the relevant columns for outlier detection
columns_to_check = ['STD']

st.write("""
    Om de data verder te cleanen hebben we gekozen voor een boxplot. Deze boxplot visualiseert de datums om te kijken of er misschien outliers zijn, door bijvoorbeeld
    een typfout tijdens het opnemen van de data. Aan de hand van de boxplot is te zien dat er geen onlogische data tussenzit.
""")

# Create box plots for each column
for column in columns_to_check:
    fig = px.box(airports_cleaned, y=column, title=f'Box Plot of {column}')
    st.plotly_chart(fig)
    st.caption('Figuur 1: Boxplot van vertrek datums')

# Maak een nieuwe kolom 'Flight_Status' op basis van ATA_ATD_ltc en STA_STD_ltc
airports_cleaned['Flight_Status'] = airports_cleaned.apply(
    lambda row: 'Op Tijd' if row['ATA_ATD_ltc'] <= row['STA_STD_ltc'] else 'Vertraagd', axis=1
)

# Laad luchthaveninformatie uit JSON
with open('airports.json') as f:
    airport_data = json.load(f)

# Functie om de breedtegraad, lengtegraad en tijdzone voor een gegeven luchthavencode uit JSON te halen
def get_airport_data(airport_code):
    for airport in airport_data:
        if airport['icao'] == airport_code:
            return airport['lat'], airport['lon'], airport['tz']
    return None, None, None

# Tab voor Schema Luchthaven
st.header("Schedule Airport")

# Toon de bovenste rijen van de schoongemaakte DataFrame inclusief Flight_Status
st.subheader("Overzicht Vluchtgegevens")
st.write("""
    Hieronder zie je een overzicht van de vluchtgegevens, inclusief geplande en werkelijke vertrektijden. 
    De dataset is opgeschoond om relevante informatie voor elke vlucht te tonen. 
    De kolom 'Flight_Status' geeft aan of een vlucht op tijd of vertraagd is.
""")
st.dataframe(airports_cleaned.head())

# Verkrijg unieke tijdzones uit de schoongemaakte dataset op basis van luchthavencodes in Org/Des
timezones = set()
for code in airports_cleaned['Org/Des'].unique():
    _, _, tz = get_airport_data(code)
    if tz:
        timezones.add(tz)

# Dropdown voor het selecteren van tijdzone
selected_timezone = st.selectbox("Selecteer Tijdzone", sorted(timezones))

# Filter de schoongemaakte dataset om alleen luchthavens uit de geselecteerde tijdzone te tonen
filtered_airports = [
    code for code in airports_cleaned['Org/Des'].unique() if get_airport_data(code)[2] == selected_timezone
]
filtered_data = airports_cleaned[airports_cleaned['Org/Des'].isin(filtered_airports)]

# Maak een kaart gecentreerd op de coördinaten van de eerste luchthaven
if not filtered_data.empty:
    first_airport_code = filtered_data['Org/Des'].iloc[0]
    lat, lon, _ = get_airport_data(first_airport_code)

    if lat and lon:
        # Maak een zwart-wit kaart
        mymap = folium.Map(location=[lat, lon], zoom_start=5, tiles='cartodb positron')
    else:
        st.error("Coördinaten voor de geselecteerde luchthaven konden niet worden gevonden.")
        mymap = folium.Map(location=[0, 0], zoom_start=2, tiles='cartodb positron')

    # Groepeer vluchtstatussen per luchthaven
    summary = filtered_data.groupby('Org/Des').agg(
        total_flights=('FLT', 'count'),
        delayed_flights=('Flight_Status', lambda x: (x == 'Vertraagd').sum()),
        ontime_flights=('Flight_Status', lambda x: (x == 'Op Tijd').sum())
    ).reset_index()

    # Bereken het percentage op tijd en wijs kleuren toe
    summary['on_time_percentage'] = (summary['ontime_flights'] / summary['total_flights']) * 100
    summary['color'] = summary['on_time_percentage'].apply(lambda x: 'green' if x >= 80 else 'red')

    # Voeg markers toe voor elke luchthaven in de geselecteerde tijdzone
    for index, row in summary.iterrows():
        lat, lon, _ = get_airport_data(row['Org/Des'])
        if lat and lon:
            # Popup om geaggregeerde vluchtgegevens weer te geven
            popup_text = (f"<div style='width: 200px;'>"
                          f"<strong>Luchthaven:</strong> {row['Org/Des']}<br>"
                          f"<strong>Totaal Aantal Vluchten:</strong> {row['total_flights']}<br>"
                          f"<strong>Vertraagde Vluchten:</strong> {row['delayed_flights']}<br>"
                          f"<strong>Op Tijd Vluchten:</strong> {row['ontime_flights']}<br>"
                          f"<strong>Percentage Op Tijd:</strong> {row['on_time_percentage']:.2f}%<br>"
                          f"</div>")

            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color=row['color']),
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(mymap)

    # Toon de kaart in Streamlit met st_folium
    st_folium(mymap, width=700, height=500)
    st.caption('Figuur 2: Visualisatie Vliegvelden')
    # Streamlit legenda
    legend_html = """
    <div style="background-color: rgba(255, 255, 255, 0.8); padding: 10px; border-radius: 5px; border: 1px solid gray; max-width: 250px;">
        <strong>Legenda Vluchtstatus</strong><br>
        <span style="color: green;">&#11044; 80% of Meer Op Tijd</span><br>
        <span style="color: red;">&#11044; Minder Dan 80% Op Tijd</span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    # Voeg een scheidingslijn toe
    st.write("---")

else:
    st.warning("Geen luchthavens gevonden voor de geselecteerde tijdzone.")

# Tweede deel: Visualisatie van Vliegtuiggegevens per Luchthavens
st.header("Visualisatie van Vliegtuiggegevens per Luchthaven")

# Laad de luchthavengegevens van CSV
airports_cleaned = pd.read_csv('schedule_airport.csv')
airports_cleaned = airports_cleaned.dropna(subset=['Org/Des'])

# Converteer de kolommen STD en STA_STD_ltc naar datetime
airports_cleaned['Flight_Time'] = pd.to_datetime(
    airports_cleaned['STD'] + ' ' + airports_cleaned['STA_STD_ltc'],
    format='%d/%m/%Y %H:%M:%S'
)

# Haal jaar en maand op voor filtering
airports_cleaned['Year'] = airports_cleaned['Flight_Time'].dt.year
airports_cleaned['Month'] = airports_cleaned['Flight_Time'].dt.month_name()  # Verkrijg maandnamen

# Maak een Year-Month kolom voor de x-as en converteer naar string
airports_cleaned['Year-Month'] = airports_cleaned['Flight_Time'].dt.to_period('M').astype(str)

st.write("""
    Deze grafiek visualiseert vluchtgegevens van verschillende luchthavens. 
    Je kunt een luchthaven selecteren en de jaren kiezen waarvan je de vluchtgegevens wilt bekijken.
    
    Na het maken van je selectie, wordt het totale aantal vluchten weergegeven. 
    Als er vluchten zijn voor de geselecteerde luchthaven en jaren, 
    wordt er een interactieve grafiek weergegeven die het aantal vluchten per maand laat zien.
    
    Deze visualisatie helpt om trends in vluchtdata te begrijpen, 
    zoals seizoensgebonden variaties in het aantal vluchten.
""")

# Dropdown voor het selecteren van luchthavens
selected_airport = st.selectbox("Selecteer Luchthaven:", airports_cleaned['Org/Des'].unique())

# Filter op Jaar
years = airports_cleaned['Year'].unique()
selected_years = st.multiselect("Selecteer Jaren:", years, default=list(years))  # Standaard op alle jaren

# Filter de gegevens op basis van de selecties
filtered_data = airports_cleaned[
    (airports_cleaned['Org/Des'] == selected_airport) &
    (airports_cleaned['Year'].isin(selected_years))
]

# Bereken het totaal aantal vluchten
total_flights = filtered_data.shape[0]
st.write(f"Totaal aantal vluchten voor {selected_airport} in geselecteerde jaren: {total_flights}")

# Als er vluchten zijn om te visualiseren
if total_flights > 0:
    # Groepeer de gegevens per Year-Month en tel vluchten voor elke maand
    flights_per_month = (
        filtered_data.groupby(['Year-Month'])
        .size()
        .reset_index(name='Aantal Vluchten')
    )

    # Maak een interactieve lijn grafiek met Plotly
    fig = px.line(
        flights_per_month,
        x='Year-Month',
        y='Aantal Vluchten',
        title=f'Aantal Vluchten per Maand voor {selected_airport}',
        markers=True  # Voegt markers toe aan de lijn voor betere zichtbaarheid
    )

    # Voeg as-labels toe
    fig.update_layout(
        xaxis_title="Jaar-Maand",
        yaxis_title="Aantal Vluchten",
        xaxis_tickformat="%Y-%m"  # Formatteer de x-as ticks om Jaar-Maand weer te geven
    )

    # Toon de Plotly-grafiek in Streamlit
    st.plotly_chart(fig)
    st.caption('Figuur 3: Lijngrafiek vliegtuigen op de Airport')
    # Voeg een scheidingslijn toe
    st.write("---")

   
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import openpyxl

# Stap 1: Dataset met vluchten inladen
df = pd.read_csv('schedule_airport.csv')
df['STA_STD_ltc'] = pd.to_datetime(df['STA_STD_ltc'], format='%H:%M:%S', errors='coerce')
df['ATA_ATD_ltc'] = pd.to_datetime(df['ATA_ATD_ltc'], format='%H:%M:%S', errors='coerce')
df['Delay'] = (df['ATA_ATD_ltc'] - df['STA_STD_ltc']).dt.total_seconds() / 60
df['Airline_IATA'] = df['FLT'].str[:2]  # Extract IATA code from first 2 characters of FLT
df['Airport'] = df['Org/Des']

# Stap 2: Luchthavens inladen
airport_data = pd.read_csv('icao_codes.csv', delimiter=';')
airport_data = airport_data[['Name', 'ICAO']]  # Houd alleen relevante kolommen
airport_data.columns = ['Name', 'Airport']  # Herbenoem de kolommen

# Stap 3: Identificeer luchthavens met gegevens in de vlucht dataset
valid_airports = df['Airport'].unique()  # Luchthavens in de vluchtgegevens
airport_data = airport_data[airport_data['Airport'].isin(valid_airports)]  # Filter de luchthavens

# Stap 4: Merge de twee datasets op luchthaven
merged_df = pd.merge(df, airport_data, left_on='Airport', right_on='Airport', how='left')

# Stap 5: Inlezen van de Excel-bestand met IATA en Airline naam
airline_data = pd.read_excel('airline_codes.xlsx')  # Update met je eigen bestandsnaam
airline_data.columns = ['IATA', 'Airline']  # Zorg dat de kolommen correct zijn benoemd

# Stap 6: Merge vluchten dataset met de airline data om de volledige airline naam te krijgen
merged_df = pd.merge(merged_df, airline_data, left_on='Airline_IATA', right_on='IATA', how='left')

# Streamlit applicatie
st.title("Gemiddelde Vertragingen per Luchtvaartmaatschappij")

st.write("""
    Deze visualisatie toont de gemiddelde vertragingen van luchtvaartmaatschappijen per luchthaven.
    Via een dropdown kun je een luchthaven selecteren, waarna een interactieve balkgrafiek de gemiddelde vertraging in minuten per airline weergeeft.
    
    De balken zijn rood voor positieve vertragingen (te laat) en groen voor negatieve vertragingen (te vroeg), met een stippellijn die nul vertraging markeert.
""")

# Combobox (typed dropdown) maken met luchthavennamen
airport_name = st.selectbox("Selecteer een luchthaven:", airport_data['Name'].unique())

# Functie om de data te filteren en de airlines te analyseren
def filter_by_airport(airport_name):
    filtered_df = merged_df[merged_df['Name'] == airport_name]
    airline_delays = filtered_df.groupby('Airline')['Delay'].mean().sort_values()

    if not airline_delays.empty:
        # Convert the values to two decimal places
        airline_delays = airline_delays.round(2)

        # Plotly barchart maken
        fig = go.Figure(data=[
            go.Bar(
                x=airline_delays.index,  # Volledige airline namen
                y=airline_delays.values,  # Vertragingen in minuten
                marker_color=['red' if val > 0 else 'green' for val in airline_delays.values],  # Kleur
                text=[f'{val:.2f}' for val in airline_delays.values],  # Format to 2 decimal places
                textposition='auto',
            )
        ])

        # Nul-lijn toevoegen
        fig.update_layout(
            shapes=[
                dict(
                    type="line",
                    x0=-0.5,  # Begin van de x-as
                    x1=len(airline_delays)-0.5,  # Eind van de x-as
                    y0=0,
                    y1=0,
                    line=dict(
                        color="black",
                        width=2,
                        dash="dashdot",  # Streepjeslijn
                    ),
                )
            ]
        )

        # Layout aanpassen voor automatische y-as schaal
        fig.update_layout(
            title=f'Gemiddelde vertragingen per luchtvaartmaatschappij voor {airport_name}',
            xaxis_title='Luchtvaartmaatschappij',
            yaxis_title='Gemiddelde vertraging (minuten)',
            yaxis=dict(
                autorange=True,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                gridcolor='lightgrey'  # Lichte gridkleur
            ), 
            plot_bgcolor='rgba(0, 0, 0, 0)',
            height=600  # Grotere grafiek
        )

        st.plotly_chart(fig)
        st.caption('Figuur 4: Vertragingen Luchtvaartmaatshappijen per Airport')
    else:
        st.write("Geen vertragingen gevonden voor deze luchthaven.")

# Roep de filter functie aan als een luchthaven is geselecteerd
if airport_name:
    filter_by_airport(airport_name)

#Alle packages die nodig zijn voor dit project.
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

#Inladen van data, ook kleine opschonen met name met behlp van pd.to_datetime. 
df = pd.read_csv('schedule_airport.csv')
df.rename(columns={'Org/Des': 'ICAO'}, inplace=True)

df2 = pd.read_csv('airports-extended-clean (1).csv', delimiter=";")
airports = df.merge(df2, on='ICAO')

airports['STA_STD_ltc'] = pd.to_datetime(airports['STA_STD_ltc'], format='%H:%M:%S')
airports['ATA_ATD_ltc'] = pd.to_datetime(airports['ATA_ATD_ltc'], format='%H:%M:%S')

#Maak een kolom aan, waarmee je de vertraging berekent.
airports['Verschil_Minuten'] = ((airports['ATA_ATD_ltc'] - airports['STA_STD_ltc']).dt.total_seconds()) / 60

#Opschoning data, niet perse nodig, maar vooral handig voor data analyse.
airports_clean = airports.drop(['ATA_ATD_ltc', 'STA_STD_ltc', 'Source'], axis=1)
airports_clean = airports_clean.rename(columns={
    'STD': 'Datum', 
    'FLT': 'Vluchtnummer', 
    'LSV': 'Inbound_Outbound', 
    'TAR': 'Geplande_gate', 
    'GAT': 'Werkelijke_gate', 
    'ACT': 'Vliegtuig_type', 
    'RWY': 'Landing_startbaan', 
    'ICAO': 'Bestemming_afkomst'
})

#Begin met maken van titel.
st.divider()
st.title("Analyse van vluchten")

#Tekst bijbehorend bij introductie
st.subheader('Data')
text = '''Vanuit de vluchten zelf hebben we gegevens gekregen. Met die gegevens hebben we een extra variabele verschil minuten gemaakt, waarmee
we de vertraging van elke vlucht hebben. Hiermee zijn we verder gegaan en hebben we zowel een bar- als boxplot gemaakt. Voor deze dataset zijn ook de landen
gebruikt, de data en de tijdzones.
'''
st.markdown(text)


#Stukje tekst die hoort bij barplot
st.header('Barplot vertraging')
text = '''Om de vertraging verder goed te visualiseren, is er eerst gekeken naar een barplot. Deze barplot is uitgesorteerd over de verschillende tijdzones,
om te kijken of er per tijdzone een groot verschil is (zie figuur 1). Er is geen significant verschil per tijdzone, maar er zijn wel opvallende waardes.
Zo is er bij Kameroen maar liefst een gemiddelde vertraging van 2 uur(!) is. Ook komen vluchten gemiddeld bij Chili 30 minuten te vroeg, wat ook niet goed is,
want er moeten buffers (reserves voor landingsbanen etc.) blijven en die buffers worden daardoor sneller opgevuld. Een verklaring van waarom deze verschillen 
van vertraging zo groot zijn, is niet gevonden en vereist verder onderzoek.
'''
st.markdown(text)
#####
#Hiermee wordt een dropbox gemaakt, waarmee is opgelet dat de tijdzone oplopend is.
airports_clean['Timezone'] = pd.to_numeric(airports_clean['Timezone'], errors='coerce')
airports_clean = airports_clean.dropna(subset=['Timezone'])
timezones_sorted = sorted(airports_clean['Timezone'].unique())
selected_timezone = st.selectbox("Selecteer tijdzone (UTC):", options=timezones_sorted)
#Gebruikt de data bijbehorend bij de tijdzone
filtered_airports = airports_clean[airports_clean['Timezone'] == selected_timezone]

#Hiermee wordt de vertraging van hoogst naar laagst gesorteerd.
landenvertraging = filtered_airports.groupby('Country')['Verschil_Minuten'].agg(np.mean)
landenvertraging_sorted = landenvertraging.sort_values(ascending=False)
rounded_text = [f"{country}: {int(round(val))} minuten" for country, val in zip(landenvertraging_sorted.index, landenvertraging_sorted.values)]
colors = ['red' if val > 0 else 'green' for val in landenvertraging_sorted.values]


#Hieronder een barplot. Opgelet moet worden is dat de data al eerst gemaakt is zodat je met .index en .values werkt.
fig_bar = go.Figure(data=[
    go.Bar(
        x=landenvertraging_sorted.index,
        y=landenvertraging_sorted.values,
        text=rounded_text,
        hoverinfo='text',
        marker=dict(color=colors)
    )
])

#Voeg tekst en dergelijke toe.
fig_bar.update_layout(
    title=f"Gemiddelde vertraging per land (minuten) in {selected_timezone} UTC",
    xaxis_title="Landen",
    yaxis_title="Gemiddelde vertraging (minuten)",
    xaxis_tickangle=-90,
    xaxis=dict(
        tickmode='array',
        tickvals=list(range(len(landenvertraging_sorted.index))),
        ticktext=landenvertraging_sorted.index
    ),
    margin=dict(l=40, r=40, t=40, b=150),
    height=600,
)
st.plotly_chart(fig_bar)
st.caption('Figuur 5: Een barplot van verschillende landen met hun gemiddelde vertraging gesorteerd per tijdzone.')
#####
# Tekst bijbehorend bij figuur 2 (boxplot)
st.header("Boxplot van vertragingen per maand per gekozen jaar.")
text = '''Om een beter beeld te krijgen van de vertragingen, is er ook gekeken of de vertragingen afhankelijk zijn van de tijd in het jaar. Zo is de verwachting
dat het drukker is in de zomer en daarom de vertragingen toch gemiddeld iets hoger zijn dan in de winter. Dit werd onderzocht met behulp van een boxplot, waarmee
de gemiddelde vertragingen per maand worden uitgebeeld per jaar (2019 en 2020). Opvallend is dat er in 2019 wel een kleine piek van vertraging lijkt te zitten in de 
zomermaanden zoals Juli, maar dit is 2020 niet het geval is. Dit valt te verklaren vanwege corona, waarbij het (vlieg)verkeer significant rustiger was. 
'''
st.markdown(text)
#Zorg ervoor dat de datum gebruikt kan worden voor verdere berekeningen
airports_clean['Datum'] = pd.to_datetime(airports_clean['Datum'], format='%d/%m/%Y')

#Voor dit hebben we de jaar nodig (selecteren van jaar) en maand nodig (boxplot zelf).
airports_clean['Year'] = airports_clean['Datum'].dt.year
airports_clean['Month'] = airports_clean['Datum'].dt.month_name()

years = airports_clean['Year'].unique()
selected_year = st.radio("Selecteer jaartal:", options=years)

#Alleen de data die gebruikt wordt welk jaar je kiest.
filtered_data = airports_clean[airports_clean['Year'] == selected_year]

#Alle uitschieters boven en onder de 300 minuten (5 uur) gaan weg. Gedachte is dat 5 uur teveel/te weinig onrealistisch is. Zorgt wel voor lichte bias
#in de gebruikte data...
filtered_data = filtered_data[(filtered_data['Verschil_Minuten'] >= -300) & 
                               (filtered_data['Verschil_Minuten'] <= 300)]
#Maken van boxplot zelf.
fig_box = px.box(filtered_data, x='Month', y='Verschil_Minuten', 
                  title=f'Boxplot van vertragingen per maand in {selected_year}',
                  labels={'Vertraging in minuten': 'Verschil Minuten', 'Maand': 'Month'},
                  category_orders={'Month': ['January', 'February', 'March', 'April', 'May', 
                                              'June', 'July', 'August', 'September', 'October', 
                                              'November', 'December']})
st.plotly_chart(fig_box)
st.caption('Figuur 6: Boxplot van alle vertragingen in minuten per maand.')







st.divider()
st.title('Vlucht data visualisatie')
st.divider()

st.subheader('Data')
text= '''De data die is gebruikt voor de visualisatie zijn zeven Excel bestanden die ieder een vlucht van Amsterdam naar Barcelona bevatten. 
De dataset  biedt een reeds aantal momentopname gedurende de vlucht van het vliegtuig. De tijdstippen in 30 seconden markeren het verloop, terwijl de breedte- en lengtegraden de geografische locatie vastleggen rond Schiphol. 

De koers in graden toont aan in welke richting het vliegtuig zich bewoog, en hoe deze richting veranderde tijdens de vlucht. De waarden in de dataset laten zien dat het vliegtuig verschillende kleine aanpassingen in zijn koers maakte. 
De snelheid, die tussen de 45 en 50 knopen ligt, wijst erop dat het vliegtuig langzaam bewoog, wat typisch is voor momenten zoals taxiën op de landingsbaan, het opstijgen of landen. 
Deze gegevens suggereren dat de vlucht zich waarschijnlijk afspeelde tijdens belangrijke fases dicht bij de grond, zoals de voorbereiding voor vertrek of de afronding van een landing.
 '''
st.markdown(text)

st.divider()
st.subheader('Landkaart vluchtroute')
text= '''Zeven vluchten zijn gevisualiseerd in één grafiek, waarbij ongeveer de helft van de vliegtuigen op een hoogte van 11.000 tot 14.000 meter vliegt, terwijl de andere helft tussen de 8.000 en 11.000 meter blijft. 
Slechts één vliegtuig verandert halverwege zijn vlucht van hoogte, waarbij het eerste deel van de vlucht op een hoogte tussen 8.000 en 11.000 meter plaatsvindt, en het tweede deel tussen 11.000 en 14.000 meter. '''
st.markdown(text)

path = '30Flight {}.xlsx'
dfs = [pd.read_excel(path.format(i+1)) for i in range(7)]

# Labels voor de dropdown
flight_labels = [f'Flight {i+1}' for i in range(len(dfs))]

# Selectie voor vluchten in Streamlit
selected_flights = st.multiselect(
    'Selecteer vluchten om weer te geven',
    flight_labels,
    default=flight_labels
)

# Functie om kleuren te bepalen op basis van hoogte
def altitude_color(altitude):
    if altitude < 2000:
        return 'blue'
    elif altitude < 5000:
        return 'cyan'
    elif altitude < 8000:
        return 'green'
    elif altitude < 11000:
        return 'yellow'
    elif altitude < 14000:
        return 'orange'
    else:
        return 'red'

# Maak een folium kaart
m = folium.Map(location=[48, 5], zoom_start=5)

# Voeg vluchtdata toe aan de kaart
for i, df in enumerate(dfs):
    if flight_labels[i] in selected_flights:
        df_clean = df.dropna(subset=['[3d Latitude]', '[3d Longitude]', '[3d Altitude M]'])
        latitudes = df_clean['[3d Latitude]'].values
        longitudes = df_clean['[3d Longitude]'].values
        altitudes = df_clean['[3d Altitude M]'].values

        for j in range(1, len(latitudes)):
            folium.PolyLine(
                locations=[[latitudes[j-1], longitudes[j-1]], [latitudes[j], longitudes[j]]],
                color=altitude_color(altitudes[j]),
                weight=2.5
            ).add_to(m)

# Gebruik Streamlit kolommen om de kaart en de legenda naast elkaar te plaatsen
col1, col2 = st.columns([3, 1])  # Verhouding kolommen: 3 delen voor de kaart, 1 deel voor de legenda

# Plaats de kaart in de eerste kolom
with col1:
    st_folium(m, width=700, height=500)

# Plaats de legenda in de tweede kolom
with col2:
    st.markdown("""
    <div style='background-color: #f0f0f0; padding:10px; border:1px solid grey;'>
        <h4 style='color: black;'>Legenda Altitude</h4>
        <div style='color: black;'><span style='background-color: blue; display:inline-block; width: 20px; height: 10px;'></span> < 2000m</div>
        <div style='color: black;'><span style='background-color: cyan; display:inline-block; width: 20px; height: 10px;'></span> 2000 - 5000m</div>
        <div style='color: black;'><span style='background-color: green; display:inline-block; width: 20px; height: 10px;'></span> 5000 - 8000m</div>
        <div style='color: black;'><span style='background-color: yellow; display:inline-block; width: 20px; height: 10px;'></span> 8000 - 11000m</div>
        <div style='color: black;'><span style='background-color: orange; display:inline-block; width: 20px; height: 10px;'></span> 11000 - 14000m</div>
        <div style='color: black;'><span style='background-color: red; display:inline-block; width: 20px; height: 10px;'></span> > 14000m</div>
    </div>
    """, unsafe_allow_html=True)

st.caption('Figuur 7: Landkaart met vluchtroute en Altitude in m.')

