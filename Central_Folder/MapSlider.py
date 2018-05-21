# Bokeh imports
from bokeh.io import curdoc
from bokeh.models import GeoJSONDataSource, LinearColorMapper, HoverTool
from bokeh.models import ColorBar
from bokeh.palettes import Magma256
from bokeh.models.widgets import Slider
from bokeh.layouts import layout, column
from bokeh.plotting import figure

# General Imports
from Locations import monthly_transfrom, make_df_shapefile, disease_studied

ALL_df = disease_studied()
data = monthly_transfrom(ALL_df)
data.find_mun()
data = data.monthly_municipality()
shp_data = make_df_shapefile(data)
column_list = shp_data.columns.tolist()
column_list = [e for e in column_list if e not in
               ('Municipality', 'geometry')]


# Here Starts the Bokeh App
def bokeh_map(disease):

    def slider_title(n):
        return 'Number of Incidences in ' + column_list[n]

    def tooltip(n):
        return [
            ("Municipality", "@Municipality"),
            ("Incidences", "@{" + column_list[N] + "}"),
        ]

    N = 0

    TOOLS = "pan,wheel_zoom,reset,hover,save"
    color_column = '2010-01'

    geojson = shp_data.to_json()

    if disease == 'Kinkhoest':
        high = shp_data['2012-04'].max()
    else:
        high = shp_data[column_list].max().max() - 3

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


bokeh_map(disease='Mumps')

# cd "Desktop\Deep-Learning-Infectious-Diseases\Central_Folder"
# bokeh serve --show MapSlider.py
# bokeh serve --show playslider_weighted.py
