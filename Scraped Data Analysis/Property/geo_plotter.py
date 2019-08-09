def quick_geo_plot():
    import pandas as pd
    """Import Data"""
    file_i = '/Users/Derrick-Vlad-/Desktop/Output_Trail_1/Clean/v1.1/' + 'Quick_Geoplot_Filtered_LATEST.csv'
    df = pd.read_csv(file_i, sep=',', encoding='unicode_escape')
    """Filter Data"""
    #df = df[df['latitude'] != 'UnableToFind']
    #df = df[(df['Latitude'].astype(float) < 2.0) & (df['Longitude'].astype(float) < 104.0)]
    print(df[['Latitude', 'Longitude']].head())
    df = df[['Latitude', 'Longitude', 'Price ($)']]
    df = df.convert_objects(convert_numeric=True).dropna()
    print(df['Latitude'].dtype)
    print(df['Longitude'].dtype)
    print(df['Price ($)'].dtype)
    """Base Map"""
    import folium
    from folium.plugins import HeatMap
    def generateBaseMap(default_location=[1.3521, 103.8198], default_zoom_start=12):
        base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
        return base_map
    base_map = generateBaseMap()
    """Prep Data"""
    d = df[['Longitude', 'Latitude', 'Price ($)']].values.tolist()
    from folium import plugins
    """Customize Heat-Map"""
    # This
    gradient = {.6: 'blue', .98: 'lime', 1: 'red'} # Original
    # Or This
    # import branca.colormap as cm
    # steps = 10
    # color_map = cm.LinearColormap('RGB').scale(0, 1).to_step(steps)
    # gradient = {}
    # for i in range(steps):
    #     gradient[1 / steps * i] = color_map.rgb_hex_str(1 / steps * i)
    max_amount = df['Price ($)'].max()
    m = plugins.HeatMap(d,
                        radius=11, min_opacity=0.1, max_zoom=1, blur=12,
                        max_val=max_amount, gradient=gradient).add_to(base_map)

    """Customize LEGEND"""
    from branca.element import Template, MacroElement
    template = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>jQuery UI Draggable - Default functionality</title>
      <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

      <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
      <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>

      <script>
      $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

      </script>
    </head>
    <body>


    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
         border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>

    <div class='legend-title'>Legend (draggable!)</div>
    <div class='legend-scale'>
      <ul class='legend-labels'>
        <li><span style='background:red;opacity:0.7;'></span>High Priced</li>
        <li><span style='background:lime;opacity:0.7;'></span>Medium Priced</li>
        <li><span style='background:blue;opacity:0.7;'></span>Relatively Low Priced</li>

      </ul>
    </div>
    </div>

    </body>
    </html>

    <style type='text/css'>
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    macro = MacroElement()
    macro._template = Template(template)
    m.get_root().add_child(macro)

    m.save('Geo_Plot.html')
    import webbrowser, os

    webbrowser.open('file://' + os.path.realpath('index.html'))

#quick_geo_plot()