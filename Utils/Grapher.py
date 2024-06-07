import matplotlib.pyplot as plt
from DataAnalysis.Data_Handler import Firm_Extractor, Load
from Utils.shared_Treatment_Grapher import map_columns_to_angles
import os
import numpy as np
from Utils.shared_Treatment_Grapher import permno_Occurences, setting_Histogram, count_Occurences_Permno, random_predictors

def Data_grapher(permnos,data):
    permno_data = Firm_Extractor(permnos,data)

    grouped_data = permno_data.groupby('permno')

    #Plotting each company's data separately
    plt.figure(figsize=(10, 6))
    for name, group in grouped_data:
        plt.plot(group['yyyymm'], group['Price'], marker='o', linestyle='-', label=f'Permno {name}')

    plt.title('Price Over Time')
    plt.xlabel('Date (yyyymm)')
    plt.ylabel('Price')
    plt.xticks(permno_data['yyyymm'][::50], rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def Polar_grapher(data, predictors_list = "Data/permno_predictors.csv"):
    #Check if the file exists
    if not os.path.isfile(predictors_list):
        raise FileNotFoundError(f"File '{predictors_list}' not found.")

    #Load permno_predictors.csv file
    permno_predictors = Load(predictors_list, show_head=False)

    #Load map predictors to angles
    predictors_angles = map_columns_to_angles()

    #find the permno list from the data
    permno_list = data['permno'].unique()

    #Start representation for radius of 1
    radius = 1

    #Setup the graph
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for permno in permno_list:
        # Find associated column names for the given permno
        associated_predictors = permno_predictors.loc[permno_predictors['permno'] == permno, 'Associated Columns'].iloc[0]

        # Convert associated_predictors string to a list of column names
        associated_predictors_list = [predictor.strip()[1:-1] for predictor in associated_predictors.strip('[]').split(',')]

        # Get angles for the associated column names
        angles = [predictors_angles[predictor_name] for predictor_name in associated_predictors_list]

        #plot these on the polar coordinates
        ax.scatter(angles, [radius] * len(angles))

        #Update the next radius
        radius += 1

    ax.set_title("polar representation", va='bottom')
    plt.show()

def Histogram_predictors(data, predictors_list = "Data/permno_predictors.csv", predictors = "Data/predictors.csv", save_path = "Graphs/Histogram_predictors occurences", nbr_predictors = 35):
 
    permno_predictors, permno_list, all_predictors_list = setting_Histogram(data, predictors_list, predictors)

    batons = count_Occurences_Permno(permno_predictors, permno_list, all_predictors_list)
 
    sorted_predictors = sorted(batons.keys(), key=lambda x: batons[x], reverse=True)
    sorted_counts = [batons[predictor] for predictor in sorted_predictors]

    plt.figure(figsize=(16, 8))
    plt.bar(np.arange(len(sorted_predictors)), sorted_counts, align='center', alpha=0.7)
    plt.axhline(y=5000, color='red', linestyle='--', linewidth=2)
    plt.xticks(np.arange(len(sorted_predictors))[::2], sorted_predictors[::2], rotation=45, ha='right', fontsize=8)
    plt.xlabel('Predictors', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Save the graph as PNG
    plt.savefig(save_path)
    plt.title('Histogram of Predictor Occurrences', fontsize=14)
    plt.show()

    return sorted_predictors[:nbr_predictors]

def time_series_plot(y_values):
    plt.figure(figsize=(10, 6))
    plt.plot( y_values, marker='o', linestyle='-')
    plt.xlabel('timeline')
    plt.ylabel('value')
    plt.xticks() 
    plt.show()

def time_series_plot_save(y_val, save_path):
    time_series_plot(y_val)
    plt.savefig(save_path)

def stock_density(data, save_path, zoom_start=None, zoom_end=None):
    step_size = 18
    #Find all dates and order them
    unique_dates = sorted(data["yyyymm"].unique().tolist())

    batons = {date: 0 for date in unique_dates}

    #Iterate over each date and count observations
    for date in data["yyyymm"]:
        batons[date] += 1

    #Convert batons dictionary to lists for plotting
    dates = list(batons.keys())
    counts = list(batons.values())

    plt.figure(figsize=(18, 8))
    plt.bar(range(len(dates)), counts, width=1, color='purple')
    plt.xlabel('Date')
    plt.ylabel('Count of Observations')
    plt.title('Histogram of Observations by Date')

    #For readability purposes
    if len(dates) > step_size:
        step = step_size
        plt.xticks(range(0, len(dates), step), dates[::step], rotation=90)
    else:
        plt.xticks(range(len(dates)), dates, rotation=90)

    if zoom_start is not None and zoom_end is not None:
        plt.xlim(zoom_start, zoom_end)

    plt.savefig(save_path, dpi=300)

    plt.show()

def predictor_appearances(data, save_path, nbr_random_pred = 15, predictors = "Data/predictors.csv"):
    all_predictors = Load(predictors, show_head=False)

    all_predictors = all_predictors['predictors'].tolist()

    random_predictors = np.random.choice(all_predictors, nbr_random_pred, replace=False)

    appearances = {predictor: [] for predictor in random_predictors}

    for date in sorted(data["yyyymm"].unique()):
        data_by_date = data[data["yyyymm"] == date][random_predictors]

        for predictor in random_predictors:
        
            count = data_by_date[predictor].notna().sum()

            appearances[predictor].append(count)

    plt.figure(figsize=(12, 6))
    for predictor, counts in appearances.items():
        plt.plot(sorted(data["yyyymm"].unique()), counts, label=predictor)

    plt.xlabel('Date')
    plt.ylabel('Count of Appearances')
    plt.xticks(rotation=90)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

    plt.title('Predictor Appearances Over Time')
    plt.show()

def Histogram_permno_occurrences(data, save_path="Graphs/Histogram_permno_occurrences.png"):

    permno_occurences = permno_Occurences(data)

    #Same as before we sort for it to be in increasing order
    sorted_permnos = permno_occurences.index
    sorted_counts = permno_occurences.values

    plt.figure(figsize=(16, 8))
    plt.bar(np.arange(len(sorted_permnos)), sorted_counts, align='center', alpha=0.7)
    plt.xlabel('Permno', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Histogram of Permno Occurrences', fontsize=14)
    plt.tight_layout()

    plt.savefig(save_path)

    plt.show()

#This plot was only done as a visualization for the oral presentation, it is not presented in the report as it makes strong
#assumptions.
def plot_cross_correlations(data, save_path):
    predictors = data.columns
    nb_of_predictors = len(predictors)

    fig, axes = plt.subplots(nb_of_predictors, nb_of_predictors, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    #Plot the cross correlations
    for i in range(nb_of_predictors):
        for j in range(nb_of_predictors):
            if i == j:
                #Distribution of the predictor on diagonal elements (if pred i and pred j are the same), we dropna which is quite restrictive.
                axes[i, j].hist(data.iloc[:, i].dropna(), bins=30, alpha=0.7)
                axes[i, j].set_title(predictors[i])
            else:
                #Cross correlation plots when pred i and pred j are different
                ccorr = data.iloc[:, i].corr(data.iloc[:, j]) #Correlation between pred i and pred j (only for the title)
                axes[i, j].scatter(data.iloc[:, i], data.iloc[:, j], alpha=0.5) #Plots pred i vs pred j -> done to observe trends
                axes[i, j].set_title(f"{predictors[i]} vs {predictors[j]}: {ccorr:.2f}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_3_random_columns(data):
    selected_predictors = random_predictors(data, 3)
    print("The predictors plotted are:", selected_predictors )
    for column in data[selected_predictors].columns:
        time_series_plot_save(y_val = (data[selected_predictors])[column], save_path = f"Graphs/Random_col_{column}.png")
        
def random_nan_repartition(data, save_path = "Graphs/nan_repartition.png"):
    selected_predictors = random_predictors(data, 3)

    total_entries_per_date = data.groupby('yyyymm').size()
    number_of_nans_per_date = data.groupby('yyyymm')[selected_predictors].apply(lambda x: x.isna().sum())

    proportion_missing_per_date = number_of_nans_per_date.div(total_entries_per_date, axis=0)

    plt.figure(figsize=(12, 8))
    for column in proportion_missing_per_date.columns:
        plt.plot(proportion_missing_per_date.index, proportion_missing_per_date[column], marker='', linestyle='-', label=column)
    plt.xlabel('Year-Month')
    plt.ylabel('Proportion of NaN Values')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.title('Proportion of NaN Values Over Time for Each Predictor')
    plt.show()
