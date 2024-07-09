import matplotlib.pyplot as plt
import numpy as np
import math
import csv

###### plot trajectory #########
def plot_trajectory(gps_path, rtab_file, c_file):
    gps_data_array = np.loadtxt(gps_path)

    gpsx_pos = []
    gpsy_pos = []
    for row in gps_data_array:
        gpsx_pos.append(float(row[0]))
        gpsy_pos.append(float(row[1]))
    gpsx_pos, gpsy_pos = np.array(gpsx_pos), np.array(gpsy_pos)


    rtab_data_array = np.loadtxt(rtab_file)
    rtabx_pos = []
    rtaby_pos = []
    for r in rtab_data_array:
        rtabx_pos.append(float(r[0]))
        rtaby_pos.append(float(r[1]))
    rtabx_pos, rtaby_pos = np.array(rtabx_pos), np.array(rtaby_pos)
    print(rtabx_pos.shape)

    c_data_array = np.loadtxt(c_file)
    cx_pos = []
    cy_pos = []
    for f in c_data_array:
        cx_pos.append(float(f[0]))
        cy_pos.append(float(f[1]))
    cx_pos, cy_pos = np.array(cx_pos), np.array(cy_pos)


    distance_error = np.sqrt((gpsx_pos-rtabx_pos)**2 + (gpsy_pos-rtaby_pos)**2)
    print("Mean Distance Error = ", distance_error.mean())

    plt.scatter(gpsx_pos, gpsy_pos, marker='D', label="GPS_synchronized", color="red", )

    plt.scatter(rtabx_pos, rtaby_pos, label="RTABmap", color="green")

    plt.scatter(cx_pos, cy_pos, label="GPS connected", s=1, color="blue")


    plt.title("Plot of Gps vs Rtabmap trajectory")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.legend(fontsize='large')
    plt.show()


############ remove duplicate rows #########
def remove_duplicate_rows_and_save(input_file_path, output_file):
    unique_rows = set()

    with open(input_file_path, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile, delimiter=' ')
        writer = csv.writer(outfile, delimiter=' ')

        for row in reader:
            # Convert the row to a tuple to make it hashable and add it to the set
            row_tuple = tuple(row)
            if row_tuple not in unique_rows:
                unique_rows.add(row_tuple)
                writer.writerow(row)



##################Insert commas #####################################
def insert_commas(input_file_path, output_file_name):
    # Read the input data
    with open(input_file_path, 'r') as input_file:
        # Read lines from the file
        lines = input_file.readlines()

    # Process each line to insert commas
    csv_data = [line.strip().replace(' ', ',') for line in lines]

    # Write the result to the output CSV file
    with open(output_file_name, 'w', newline='') as output_csv_file:
        csv_writer = csv.writer(output_csv_file)
        csv_writer.writerows([row.split(',') for row in csv_data])

    print(f"CSV file has been created: {output_file_name}.")

def txt_to_csv(input_txt_path, output_csv_path):
    # Read the data from the text file using numpy.genfromtxt
    data = np.genfromtxt(input_txt_path, dtype=None, encoding=None)

    # Save the data to a CSV file using numpy.savetxt
    np.savetxt(output_csv_path, data, delimiter=',', fmt='%s')

    print(f"CSV file has been created: {output_csv_path}.")




input_file_path = "/home/paulosuma/Documents/SafeAuto/mp-release-23fa-main/rtabdataxy.csv"
output_file_path = "/home/paulosuma/Documents/SafeAuto/mp-release-23fa-main/gpsdataxycommas.csv"


# insert_commas(input_file_path, output_file_path)

# remove_duplicate_rows_and_save(input_file_path, output_file_path)


gps_file_path = "/home/paulosuma/Documents/SafeAuto/mp-release-23fa-main/gpsdataxy.csv"
rtab_file_path = "/home/paulosuma/Documents/SafeAuto/mp-release-23fa-main/rtabdataxy.csv"
connected_gps = "/home/paulosuma/Documents/SafeAuto/mp-release-23fa-main/cdataxy.csv"

plot_trajectory(gps_file_path, rtab_file_path, connected_gps)




