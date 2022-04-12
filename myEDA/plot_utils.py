import matplotlib.pyplot as plt
import utils

def bar_chart(x, y, title, x_axis_label, y_axis_label, rotation_choice, grid_lines_on=True):
    """Shows a pyplot bar chart with titles and labels
    """
    plt.figure()
    plt.bar(x, y, color="orange")
    #titles
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    #rotate xticks
    plt.xticks(rotation=rotation_choice)
    plt.grid(grid_lines_on)
    plt.show()

def pie_chart(x, y, title):
    """Shows a pyplot pie chart with titles and labels as percents with 2 decimal places
    """
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.title(title)
    plt.show()

def discretized_bar(x, y, bar_width, title, x_axis_label, y_axis_label, binlocs, binlabels):
    """Shows a pyplot bar chart with titles, labels, and specified bin widths and xtick locations
    """
    plt.figure()
    plt.bar(x, y, width=bar_width, edgecolor="black", align="edge")
    #titles
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    #tick labels
    plt.xticks(ticks=binlocs, labels=binlabels, rotation=45)
    plt.show()

def histogram(data, title, x_axis_label, y_axis_label):
    """Shows a pyplot histogram with titles and labels
    """
    plt.figure()
    plt.hist(data)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.show()

def linear_regression_plot(x, y, m, b, cor_label, title, x_axis_label, y_axis_label):
    """Shows a pyplot scatter plot with least squares best fit line using y=mx+b, annotates plot with r and covariance
    """
    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    #least squares line fit plot:
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="r")
    plt.annotate(cor_label, xy=(1, 1), xycoords="figure fraction", horizontalalignment="center", color="red")
    plt.show()

def box_plot(distributions, labels, title, x_axis_label, y_axis_label, rotation="horizontal"):
    """Shows a pyplot boxplot of distributions with titles and labels and customized xtick labels
    """
    plt.figure()
    plt.boxplot(distributions)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    
    # customize x ticks
    plt.xticks(list(range(1, len(distributions) + 1)), labels, rotation=rotation)
    
    plt.show()