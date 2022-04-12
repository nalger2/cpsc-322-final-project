"""
This is a utility functions file with reusable functions I have 
implemented in the various jupyter notebooks for PA3
"""
import numpy as np
import math
def get_column(table, header, col_name):
    """Reusable function accesses a column in a given table based on the column name

    Args:
        table (list list): table of data
        header(list str): column header names
        col_name(str): column name to access and return

    Returns:
        column(list): the desired column in list form
    """
    col_index = header.index(col_name)
    column = []
    for row in table:
        value = row[col_index]
        if value not in ('NA', 'N/A', ''):
            column.append(value)
    return column

def get_frequencies(table, header, col_name):
    """This reusable function gathers the frequencies of a NUMERICAL attribute column

    Args:
        table (list list): table of data
        header(list str): column header names
        col_name(str): column name to find frequencies

    Returns:
        values(list): list of the various values found in the column
        counts(list int): list of the counts for each of the values in values (parallel list)
    """
    col = get_column(table, header, col_name)
    #this method only works for numeric values
    #to do this with nominal attributes, use a dictionary
    col.sort() #in place sort
    values = []
    counts = []
    for val in col:
        if val in values:
            counts[-1] += 1
        else:
            values.append(val)
            counts.append(1)
    return values, counts

def get_frequencies_nominal(table, header, col_name):
    """This reusable function gathers the frequencies of a NOMINAL attribute column,
    returns a dictionary

    Args:
        table (list list): table of data
        header(list str): column header names
        col_name(str): column name to find frequencies

    Returns:
        freq(dict): dictionary of values with their given frequencies
    """
    col = get_column(table, header, col_name)
    freq = {}
    for name in col:
        name = str(name)
        if (name in freq):
            freq[name] += 1
        else:
            freq[name] = 1
    return freq #returns frequencies dictionary

def get_col_sums(table, header, col_names):
    """This reusable function gets a list of column sums for given column names

    Args:
        table (list list): table of data
        header(list str): column header names
        col_names(list str): column names for desired sums

    Returns:
        sums_list(list int): list of column sums in a parallel list to the given column names
    """
    sums_list = []
    for colname in col_names:
        col = get_column(table, header, colname)
        sums_list.append(round(sum(col), 2))
    return sums_list

def confirm_attribute_is_sum_others(table, header, rows, col_name_sum, col_parts):
    """This reusable functionconfirms (asserts) one attribute of a list of rows is the
    sum of a list of other attributes

    Args:
        table (list list): table of data
        header(list str): column header names
        rows(list int): list of row indexes to check
        col_name_sum(str): name of column to check is the sum of col_parts columns
        col_parts(list str): list of names of columns to add up to col_name_sum
    """
    for row in rows:
        sum_index = header.index(col_name_sum)
        sum_list = []
        for colpart in col_parts:
            part_index = header.index(colpart)
            sum_list.append(table[row][part_index])
        part_sum = sum(sum_list)
        assert np.isclose(part_sum, table[row][sum_index])
        print("Confirmed Row", row, col_name_sum, "is a sum of", col_parts)

def create_ratings_column(auto_table):
    """Dataset-specific function to create a fuel economy ratings column based on mpg column

    Args:
        auto_table (myPyTable): auto data mypytable object
    
    Returns:
        auto_table (myPyTable): auto data mypytable object with ratings column added
    """
    mpg_index = auto_table.column_names.index("mpg")
    auto_table.column_names.append("fuel economy rating")
    for row in auto_table.data:
        mpg = row[mpg_index]
        if mpg <= 13:
            rating = 1
        elif mpg == 14:
            rating = 2
        elif mpg <= 16:
            rating = 3
        elif mpg <= 19:
            rating = 4
        elif mpg <= 23:
            rating = 5
        elif mpg <= 26:
            rating = 6
        elif mpg <= 30:
            rating = 7
        elif mpg <= 36:
            rating = 8
        elif mpg <= 44:
            rating = 9
        else:
            rating = 10
        row.append(rating)
    return auto_table

def compute_equal_width_cutoffs(values, num_bins):
    """Reusable function- computes bin cutoffs with equal widths according
    to the given values, and number of bins

    Args:
        values(list int or float): list of values to use for bin computations
        num_bins(int): number of desired equal-width bins

    Returns:
        cutoffs(list float): list of cutoff points, bins+1 number of cutoffs
    """
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins #produces float - the width of each bin
    cutoffs = list(np.arange(min(values), max(values), bin_width)) #for floating point start stop and steps
    cutoffs.append(max(values)) #append the last cutoff - max
    #convert cutoffs to ints, or round them
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs

def compute_bin_frequencies(values, cutoffs):
    """Reusable function computes the frequencies for each bin
    based on the given cutoffs and values

    Args:
        values (list int or float): list of values to be placed into frequency bins
        cutoffs (list float): list of cutoff points for the bins
    
    Returns:
        freqs(list int): list of frequencies per bin
    """
    freqs = [0 for _ in range(len(cutoffs) - 1)] #a list of [0,0,0,0,0] length is cutoffs - 1 (there are n+1 cutoffs)
    for value in values:
        if value == max(values): #last bin is fully closed - handling weird case
            freqs[-1] += 1 #increment last bin's freq for max value case
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i+1]: #left side closed, right side open
                    freqs[i] += 1
    return freqs

def bins_details(cutoffs):
    """Reusable function gathers details for bins: width of each bin, 
    the locations for x tick labels, and the x-tick labels themselves

    Args:
        cutoffs(list float): bin cutoff locations
    
    Returns:
        binwidth(float): width of each bin
        bin_label_locs(list float): location for bin's x-tick labels (middle of each bin)
        bin_labels(list str): bin labels for bin's xtick labels
    """
    binwidth = cutoffs[1] - cutoffs[0]
    bin_label_locs = []
    bin_labels = []
    for i in range(len(cutoffs) - 1):
        loc = round(cutoffs[i] + binwidth / 2, 2)
        bin_label_locs.append(loc)
        bin_labels.append(str(int(cutoffs[i])) + "--" + str(int(cutoffs[i + 1])))
    return binwidth, bin_label_locs, bin_labels

def compute_linear_regression_stats(x, y):
    """ Computes the slope (m) and intercept (b) for the formula y = mx + b

    Arguments:
        x (list int or float):
        y (list int or float):
    
    Returns:
        m (float): slope of the line
        b (float): intercept of the line
    """

    meanx = sum(x)/len(x)
    meany = sum(y)/len(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    m_den = sum([(x[i] - meanx) ** 2 for i in range(len(x))])
    m = num / m_den 
    
    r_den = math.sqrt(sum([(x[i] - meanx) ** 2 for i in range(len(x))]) * sum([(y[i] - meany) ** 2 for i in range(len(y))]))
    r = num / r_den

    cov = num // len(x)
    # y = mx + b => b = y - mx 
    b = meany - m * meanx

    return m, b, r, cov

def pct_strings_to_numeric(list_pcts):
    """Converts a list of percent strings formatted as "98%" into a list of floats, i.e. 98.0
    
    Args:
        list_pct(list str): list of percents as strings

    Returns:
        numeric_pcts(list float): parallel list of floating point percent values
    """
    numeric_pcts = []
    for pct in list_pcts:
        num_pct = float(pct[:-1])
        numeric_pcts.append(num_pct)
    return numeric_pcts

def genre_ratings_counts(genres, ratings):
    """Makes parallel lists of movie ratings scored by genre
    
    Args:
        genres: 2D list of genres by movie instance
        ratings: 1D list of ratings by movie instance

    Returns:
        list_of_genres: parallel to ratings list, list of all genre names
        rating_by_genre: parallel to list of genres, ratings for each genre
    """
    list_of_genres = []
    ratings_by_genre = []
    for i in range(len(genres)): #a 2d list of each instance's multiple genres
        for genre in genres[i]:
            if genre not in list_of_genres:
                list_of_genres.append(genre)
                ratings_by_genre.append([ratings[i]])
            else:
                this_gen_index = list_of_genres.index(genre)
                ratings_by_genre[this_gen_index].append(ratings[i])
    return list_of_genres, ratings_by_genre
                
            






def group_by(table, header, groupby_col_name): #***HAVENT USED
    groupby_col_index = header.index(groupby_col_name)
    groupby_col = get_column(table, header, groupby_col_name)
    group_names = sorted(list(set(groupby_col)))
    groupby_subtables = [[] for _ in group_names] #e.g. [[],[],[]]

    for row in table:
        groupby_val = row[groupby_col_index] #this row's value at this index
        groupby_val_subtable_index = group_names.index(groupby_val)
        groupby_subtables[groupby_val_subtable_index].append(row.copy()) #make a copy

    return group_names, groupby_subtables
"""
def true_column_counts(table, header, list_column_names):
    true_counts = [0 for i in range(len(list_column_names))] #parallel to list_column_names
    for i in range(len(list_column_names)):
        col_name = list_column_names[i]
        col_index = header.index(col_name)
        for row in table:
            if row[col_index] == True:
                true_counts[i] += 1
    return true_counts
"""