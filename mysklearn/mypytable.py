"""myPyTable defines the class MyPyTable
"""
import copy
import csv
from tabulate import tabulate
#from mysklearn import myutils


class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
        *TODO insert class-level attributes here (if any???)
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        if isinstance(col_identifier, str): #if col identifier passed in as a string, get index
            col_identifer = self.get_col_index(col_identifier)
        col = []
        for row in self.data:
            col.append(row[col_identifer])
        if not include_missing_values:
            col = [value for value in col if value not in ("NA", "N/A", '')]
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for i in range(len(row)):
                try:
                    numeric_val = float(row[i]) #might fail, so TRY
                    row[i] = numeric_val
                except ValueError:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop.sort(reverse=True) #prevent index errors by going from end of table up
        for index in row_indexes_to_drop:
            self.data.pop(index)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, "r") as file:
            reader = csv.reader(file)
            data = []
            for row in reader:
                data.append(row)
        header = data.pop(0)
        self.data = data
        self.column_names = header
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        with open(filename, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns
            list of int: list of indexes of duplicate rows found
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        key_indexes = []
        #convert col names to indexes
        for name in key_column_names:
            key_indexes.append(self.get_col_index(name))
        list_of_indexes = []
        #outer loop (entire list)
        for i in range(len(self.data)):
            curr_row = self.data[i] #row to compare 1
            #inner loop (i + 1 and on)
            for j in range(i + 1,len(self.data)):
                compare_row = self.data[j] #row to compare 2

                flag = True #row is duplicate until proven otherwise
                for ind in key_indexes:
                    if curr_row[ind] != compare_row[ind]:
                        flag = False #not a duplicate
                if flag is True and (j not in list_of_indexes): #only add duplicates once
                    list_of_indexes.append(j)
        return list_of_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        rows_to_remove = []
        for i in range(len(self.data)):
            for att in self.data[i]:
                if att in ("NA", "N/A", ''):
                    if i not in rows_to_remove:
                        rows_to_remove.append(i)
        rows_to_remove.sort(reverse=True)
        for row in rows_to_remove:
            self.data.pop(row)

    def remove_rows_with_missing_values_by_col(self, col_name):
        """Remove rows from table data with missing values only in given column name
        Args:
            col_name(str): name of column to search for missing values
        """
        col_index = self.get_col_index(col_name)
        rows_to_remove = []
        for i in range(len(self.data)):
            if self.data[i][col_index] in ("NA", "N/A", ''):
                rows_to_remove.append(i)
        rows_to_remove.sort(reverse=True)
        for row in rows_to_remove:
            self.data.pop(row)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.get_col_index(col_name)
        all_column = self.get_column(col_name, include_missing_values=False)
        avg = sum(all_column) / len(all_column)
        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        stats_headers = ["attribute", "min", "max", "mid", "avg", "median"]
        stats_data = []
        stats_table = MyPyTable(column_names=stats_headers, data=stats_data)
        if len(col_names) >= 1: #if list not empty
            for name in col_names:
                col_to_compute = self.get_column(name, include_missing_values=False)
                if len(col_to_compute) >= 1: #if column has data
                    stats_row = [] #row to add to data table
                    stats_row.append(name) #attribute name
                    stats_row.append(min(col_to_compute))
                    stats_row.append(max(col_to_compute))
                    stats_row.append(self.find_mid(col_to_compute))
                    stats_row.append(self.find_avg(col_to_compute))
                    stats_row.append(self.find_median(col_to_compute))

                    stats_data.append(stats_row) #add row to data table
        stats_table.data = stats_data #add data table to object
        return stats_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """
        #join headers of self and other table
        new_headers = [self.column_names]
        new_headers += [name for name in other_table.column_names if name not in self.column_names]

        #t1 refers to self, t2 refers to other table
        key_indexes_1, key_indexes_2 = self.gather_key_indexes(other_table, key_column_names)

        #join data into rows
        joined_data = []
        for i in self.data: #for row in t1
            for j in other_table.data: #for row in t2
                if self.check_rows_match(i,j,key_indexes_1, key_indexes_2):
                    new_row = self.create_joined_row(i, j, other_table, new_headers)
                    joined_data.append(new_row) #builds data table
        joined_table = MyPyTable(new_headers,joined_data) #builds mypytable with new headers/table
        return joined_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        #join headers of self and other table
        new_headers = [self.column_names]
        new_headers += [name for name in other_table.column_names if name not in self.column_names]

        #key indexes for both tables:
        key_indexes_1, key_indexes_2 = self.gather_key_indexes(other_table, key_column_names)

        #begin loops to build outer joined table
        joined_data = []
        rows_added_to_joined = [] #solely acts as a tracker to keep track of rows added to table
        for i in self.data: #for row in t1
            match_exists = False #flag
            for j in other_table.data: #for row in t2
                if self.check_rows_match(i,j,key_indexes_1, key_indexes_2):
                    rows_added_to_joined.append(j) #keep track of rows added to joined table
                    new_row = self.create_joined_row(i, j, other_table, new_headers)
                    match_exists = True
                    joined_data.append(new_row) #builds data table

            if match_exists is False: #no match was found in other table
                #create a row with NA values
                new_row = []
                for header in new_headers:
                    if header in self.column_names:
                        val_to_insert = i[self.get_col_index(header)]
                    else:
                        val_to_insert = "NA"
                    new_row.append(val_to_insert) #builds row
                joined_data.append(new_row)

        #now add rows from  OTHER table that weren't already added
        for row in other_table.data: #for row in t2
            if row not in rows_added_to_joined: #kept track of which rows have been added already
                new_row = []
                for header in new_headers:
                    if header in other_table.column_names:
                        val_to_insert = row[other_table.get_col_index(header)]
                    else:
                        val_to_insert = "NA"
                    new_row.append(val_to_insert) #builds row
                joined_data.append(new_row)

        return MyPyTable(new_headers,joined_data) #builds mypytable with new headers/data

    #MY METHODS
    def get_col_index(self, col_name):
        """Return the index of a column given the column's name.
        Args:
            col_name(str): column's name
        Returns:
            col_index(int): column's index
        """
        try:
            col_index = self.column_names.index(col_name)
            return col_index
        except ValueError:
            print("ERROR: column name",col_name, "does not exist in this table")
            return 0

    def get_col_indexes(self, col_names):
        """Return the index of a column given the column's name.
        Args:
            col_name(str): column's name
        Returns:
            col_index(int): column's index
        """
        col_indexes = []
        for col_name in col_names:
            col_indexes.append(self.get_col_index(col_name))
        return col_indexes

    @staticmethod
    def find_mid(numlist):
        """Find the midpoint of a list, or halfway between the min and max
        Args:
            list(list): list to compute midpoint
        Returns:
            midpoint(int): midpoint
        """
        midpoint = (max(numlist) + min(numlist)) / 2
        return midpoint

    @staticmethod
    def find_avg(numlist):
        """Find the average of a list
        Args:
            list(list): list to compute average
        Returns:
            average(float): average of list
        """
        return sum(numlist) / len(numlist)

    @staticmethod
    def find_median(median_list):
        """Find the median of a list, accounting even and odd list lengths, and
        for integer division rounding
        Args:
            list(list): list to compute median
        Returns:
            median(float): median of list
        """
        median_list.sort()
        listlen = len(median_list)
        #find median based on list length:
        if len(median_list) % 2 == 1: #odd list
            median = median_list[(listlen // 2)]
        else: #even list
            pt1 = median_list[listlen // 2]
            pt2 = median_list[(listlen // 2) - 1]
            median = (pt1 + pt2) / 2
        return median

    def gather_key_indexes(self, other_table, key_column_names):
        """Finds the key indexes for both tables to be joined
        Args:
            other_table(list): 2nd table for joining
            key_column_names(list str): list of column names for indexing
        Returns:
            key_indexes_1: key indexes for table 1
            key_indexes_2: key indexes for table 12
        """
        key_indexes_1 = []
        key_indexes_2 = []
        for name in key_column_names: #ensures indexes are in the same order as the key_column_names
            key_indexes_1.append(self.get_col_index(name))
            key_indexes_2.append(other_table.get_col_index(name))
        return key_indexes_1, key_indexes_2

    @staticmethod
    def check_rows_match(list1, list2, list_key_indexes_1, list_key_indexes_2):
        """Checks if two different lists match given each list's key indexes to check
        Args:
            list1(list): left comparison
            list2(list): right comparison
            list_key_indexes_1(list int): list of list1's indexes to compare
            list_key_indexes_2(list int): list of list2's indexes to compare
        Returns:
            rows_match_bool(bool): true or false whether rows match
        """
        rows_match_bool = True
        for i in range(len(list_key_indexes_1)):
            if list1[list_key_indexes_1[i]] != list2[list_key_indexes_2[i]]:
                rows_match_bool = False
        return rows_match_bool

    def create_joined_row(self, rowt1, rowt2, other_table, headers_list):
        """Creates a joined row from two different rows with the correct order
        given a list of corresponding headers
        Args:
            rowt1(list): row from first table
            rowt2(list): row from second table
            other_table(myPyTable): other table being joined
            headers_list(list str): list of headers for joined table
        Returns:
            new_row(list): newly joined row
        """
        new_row = []
        #ensures row is in the same order as the headers
        for header in headers_list:
            if header in self.column_names:
                val_to_insert = rowt1[self.get_col_index(header)]
            elif header in other_table.column_names: #name must be in the 2nd table
                val_to_insert = rowt2[other_table.get_col_index(header)]
            new_row.append(val_to_insert) #builds row
        return new_row

    def count_missing_values(self):
        """Counts the missing values in each column
        Args:
        Returns:
            na_counts(list int): list of  missing value counts by column, parallel to self.column_names
        """
        na_counts = []
        for col_name in self.column_names:
            col = self.get_column(col_name, include_missing_values=True)
            count = col.count("N/A") + col.count("NA") + col.count("na") + col.count('')
            na_counts.append(count)
        #print info nicely
        print("Missing Values:")
        na_indexes = []
        for i in range(len(na_counts)):
            if na_counts[i] > 0:
                na_indexes.append(i)
        for index in na_indexes:
            print("\tColumn \"" + self.column_names[index] + "\": " + str(na_counts[index]))
        return na_counts

    def remove_columns(self, col_indexes):
        """
        col_indexes: list col indexes
        """
        col_indexes = sorted(col_indexes, reverse=True) #prevents removing the wrong indexes/out of range
        for col in col_indexes:
            self.column_names.pop(col)
        for row in self.data:
            for col in col_indexes:
                row.pop(col)