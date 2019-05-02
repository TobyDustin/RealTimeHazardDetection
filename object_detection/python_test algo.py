from gpxplotter import read_gpx_file
from gpxplotter.mplplotting import plot_elevation_hr_multi_dist, save_fig
from matplotlib import pyplot as plt
plt.style.use('seaborn-poster')


for track in read_gpx_file('test.gpx'):
    for i, segment in enumerate(track['segments']):
        fig = plot_elevation_hr_multi_dist(track, segment)
        save_fig(fig, 'test-{}.png'.format(i))
