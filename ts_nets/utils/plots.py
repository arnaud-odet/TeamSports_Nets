import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import seaborn as sns  
from scipy.spatial import Voronoi, voronoi_plot_2d

# Basis     

def plot_field(ax=None):
    if ax == None :
        fig, ax = plt.subplots(1,1,figsize = (6,4))
    rect = Rectangle(xy = (-15,-5), width= 130, height= 80, color = 'green', fill = True, alpha = 0.6)
    ax.add_patch(rect)
    field = [[0,0], [0,70], [100,70], [100,0]]
    for i in range(len(field)):
        ax.plot([field[i][0], field[(i+1)%len(field)][0]], [field[i][1], field[(i+1)%len(field)][1]], color = 'white');
    ax.plot([22,22], [0,70], color = 'white', linestyle = '--');
    ax.plot([50,50], [0,70], color = 'white');
    ax.plot([78,78], [0,70], color = 'white', linestyle = '--');
    ax.scatter(x=[0,0,50,100,100], y=[32,38,35,32,38], color='white', s = 6)

# Collective behavior

def plot_voronoi(df:pd.DataFrame, frame:int, ax=None):
    ax.cla()
    field_bounds = {
        'x_min': 0,
        'x_max': 100,
        'y_min': 0,
        'y_max': 70
    }
    player_coordinates = [[x,y] for x,y in zip(df.query(f"frame == {frame}")['x'], df.query(f"frame == {frame}")['y'])]
    plot_field(ax=ax);
    vor = Voronoi(player_coordinates)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points = False);

    # Limit Voronoi regions to the field boundaries
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            # Check if any point in the polygon lies outside the field boundaries
            if all(field_bounds['x_min'] <= p[0] <= field_bounds['x_max'] and
                field_bounds['y_min'] <= p[1] <= field_bounds['y_max'] for p in polygon):
                ax.fill(*zip(*polygon), alpha=0.4);
                    
    sns.scatterplot(data = df.query(f"frame == {frame}"), x = 'x', y='y', hue = 'group', ax=ax, legend= False);
    ax.set_xlim(-15,115)
    ax.set_ylim(-5,75)

    
    return ax


# ANN 

def plot_history(history, ax=None):
    if ax is None :
        fig, ax = plt.subplots(1,1,figsize = (6,4))
    
    ax.plot(np.sqrt(history.history['loss']))
    ax.plot(np.sqrt(history.history['val_loss']))
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Val'], loc='best')
    #plt.ylim((max(np.sqrt(history.history['loss']).min()-0.5,0),np.sqrt(history.history['loss']).min()+0.5))
    #plt.show()

def plot_weight_and_biases_distribution(dl_model, ax_w=None, ax_b=None) :   
    # Extract W&B from the layers
    weights = []
    biases = []

    for layer in dl_model.layers:
        if hasattr(layer, 'get_weights'):
            layer_weight_object = layer.get_weights()
            if len(layer_weight_object) > 1:  # Checking if there are biases
                for weights_set in layer_weight_object[0]:
                    weights.extend(weights_set.flatten())
                bias_values = layer_weight_object[1]
                biases.extend(bias_values.flatten())

    # Plot the distribution
    if ax_w is None or ax_b is None:
        fig, [ax_w,ax_b] =plt.subplots(1,2,figsize=(12,4))
    
    ax_w.set_title(f"Distribution of Weights");
    sns.histplot( x = weights, kde = True, ax= ax_w);
    ax_b.set_title(f"Distribution of Biases");
    sns.histplot( x = biases, kde = True, ax= ax_b);
    #plt.show()
    