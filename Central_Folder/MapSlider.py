# Bokeh imports
from bokeh.io import curdoc
from bokeh.models import GeoJSONDataSource, LinearColorMapper, HoverTool
from bokeh.models import ColorBar
from bokeh.palettes import Magma256
from bokeh.models.widgets import Slider
from bokeh.layouts import layout, column
from bokeh.plotting import figure

# General Imports to preprocess the data
from Locations import MonthlyTransfrom, make_df_shapefile, disease_studied

# Here Starts the Bokeh App
def bokeh_map():
    """This function creates the choropleth map with the time slider using Bokeh library.
    The input to this function is the geopandas DataFrame that has the 'Geometry column':
    index  Municipality  2004-01  2004-02  2004-3  ... Geometry
      0     Eindhoven       2       5       12         POLYGON()
      1       Breda         8       14      9          POLYGON()
    Obviously, the time slider will read from the columns 2004-01, 2004-02, etc.
    """

    disease = str(input('What disease would you like to Visualize? '))

    # The following condition is to make the map better since the max values
    # only correspond to just a few municipalities.

    # Data Preprocessing
    ALL_df = disease_studied()
    data = MonthlyTransfrom(ALL_df)
    data.find_mun()
    data = data.monthly_municipality()
    shp_data = make_df_shapefile(data)
    column_list = shp_data.columns.tolist()
    column_list = [e for e in column_list if e not in
                   ('Municipality', 'geometry')]

    # Map Building
    if disease == 'Kinkhoest':
        high = shp_data['2012-04'].max()
    else:
        high = shp_data[column_list].max().max() - 3

    TOOLS = "pan,wheel_zoom,reset,hover,save"
    color_column = '2010-01'

    geojson = shp_data.to_json()

    def slider_title(n):
        return 'Number of Incidences in ' + column_list[n]

    def tooltip(n):
        return [
            ("Municipality", "@Municipality"),
            ("Incidences", "@{" + column_list[N] + "}"),
        ]
    # Initial Column of the Time Slider
    N = 0

    mapper = LinearColorMapper(palette=Magma256[::-1],
                               low=0, high=high)

    geo_source = GeoJSONDataSource(geojson=geojson)
    p = figure(tools=TOOLS, toolbar_sticky=False,
               plot_width=600, plot_height=285)
    patches_renderer = p.patches('xs', 'ys', fill_alpha=0.7,
                                 fill_color={'field': color_column,
                                             'transform': mapper},
                                 line_color='black', line_width=0.2,
                                 source=geo_source)

    # This is to remove the grid lines of the map to make it look better.
    p.xgrid.visible = False
    p.ygrid.visible = False

    hover = p.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [
        ("Municipality", "@Municipality"),
        ("Incidences", "@{"+color_column+"}"),
        ]

    color_bar = ColorBar(color_mapper=mapper, label_standoff=12,
                         border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    slider = Slider(start=0, end=len(column_list)-1, value=N, step=1,
                    width=515, show_value=False, tooltips=False,
                    title='Number of Incidences in ' + column_list[N])

    def callback(attr, old, new):
        N = slider.value
        color_column = column_list[new]
        slider.title = slider_title(new)
        g = patches_renderer.glyph
        g.fill_color = {**g.fill_color, 'field': color_column}
        hover = p.select_one(HoverTool)
        hover.point_policy = "follow_mouse"
        hover.tooltips = [
            ("Municipality", "@Municipality"),
            ("Incidences", "@{" + column_list[N] + "}"),
        ]

    slider.on_change('value', callback)

    layout = column(p, slider)
    curdoc().add_root(layout)


bokeh_map()

# cd "Desktop\Deep-Learning-Infectious-Diseases\Central_Folder"
# bokeh serve --show MapSlider.py
# bokeh serve --show playslider_weighted.py
