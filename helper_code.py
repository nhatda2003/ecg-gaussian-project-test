import numpy as np
import os
import sys
import wfdb
import pickle

# Find the records in a folder and its subfolders.
#NHATedit
def find_records(folder):
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            if file.endswith('.npy'):
                filetype = file[:-4][-5:]
                if filetype == 'label':
                    #print(filetype)
                    parts = file[:-4].split('_')  # Split the filename by underscore
                    file_record = '_'.join(parts[:2])  # Join the first two parts with underscore
                    record = os.path.relpath(os.path.join(root, file_record), folder)
                    records.add(record)
                    #print(record)
                    #print("???")
                    #raise Exception ("Error")
    records = sorted(records)
    
    #print(records)
    #raise Exception ("Error")
    return records


def find_records_ptbxl(folder):
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.hea':
                record = os.path.relpath(os.path.join(root, file), folder)[:-4]
                records.add(record)
    records = sorted(records)
    return records


# Load the header for a record.
def load_header(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    return header


#NHATedit modify to read each file at once
def load_raw_data_ptbxl(sampling_rate, path):
    #print("path in load_raw_data_ptbxl:", path)
    if sampling_rate == 100:
        #if os.path.exists(path + '_raw100.npy'):
        data = np.load(path+'_raw100.npy', allow_pickle=True)
        label = np.load(path+'_label.npy', allow_pickle=True)
        # else:
        #     data = [wfdb.rdsamp(path)]
        #     #data = wfdb.rdrecord(path)
        #     #print("asdasdasdasdasdasdasdasd",data)
        #     #raise Exception("Stop check wfdb")
        #     data = np.array([signal for signal, meta in data]) #data
        #     pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    else:
        raise Exception("sampling_rate not equal to 100")
    # elif sampling_rate == 500:
    #     if os.path.exists(path + 'raw500.npy'):
    #         data = np.load(path+'raw500.npy', allow_pickle=True)
    #     else:
    #         data = [wfdb.rdsamp(path)]
    #         data = np.array([signal for signal, meta in data])
    #         pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    # if len(data) == 1: #Check if this is the original full 12 lead data or just load .npy from single lead I/II
    #     data = (data[0].T)
    #     data = data[0]   
    #NHATnote: Default: Take lead I
    return data, label
###############################################################################NHATedit


##################################NHATedit function to change the name in .hea file
def change_file_name_in_hea_file(hea_file_path, new_file_name):
    first = 0
    
    with open(hea_file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith('#'):
            
            line = line
            new_lines.append(line)
        else:
            if first == 0:
                first +=1
                parts = line.split()
                parts[0] = new_file_name
            else:
                parts = line.split()
                var = new_file_name+".dat"
                parts[0] = var  # Change the file name
            parts.append("\n")
            new_line = ' '.join(parts)

            new_lines.append(new_line)

    # Write the modified lines back to the .hea file
    with open(hea_file_path, 'w') as f:
        f.writelines(new_lines)
######################################


# Load the signal(s) for a record.
def load_signal(record):
    import wfdb
    signal_files = get_signal_files(record)
    if signal_files:
        signal, fields = wfdb.rdsamp(record)
    else:
        signal, fields = None, None
    return signal, fields

# Load the signal(s) for a record.
def load_signals(record):
    return load_signal(record)

# Load the dx class(es) for a record.
# def load_dx(record):
#     header = load_header(record)
#     dx = get_dxs_from_header(header)
#     return dx

#NHATedit new load_dx function
def load_dx(record):
    label = np.load(record+'_label.npy', allow_pickle=True)
    if label == 0:
        return ['Normal']
    if label == 1:
        return ['Abnormal']
    raise Exception("Error in loading label, not 0, not 1")


def load_dxs(record):
    return load_dx(record)

# Save the header for a record.
def save_header(record, header):
    header_file = get_header_file(record)
    save_text(header_file, header)

# Save the signal(s) for a record.
def save_signal(record, signal, comments=list()):
    header = load_header(record)
    path, record = os.path.split(record)
    sampling_frequency = get_sampling_frequency(header)
    signal_formats = get_signal_formats(header)
    adc_gains = get_adc_gains(header)
    baselines = get_baselines(header)
    signal_units = get_signal_units(header)
    signal_names = get_signal_names(header)

    if all(signal_format == '16' for signal_format in signal_formats):
        signal = np.clip(signal, -2**15 + 1, 2**15 - 1)
        signal = np.asarray(signal, dtype=np.int16)
    else:
        signal_format_string = ', '.join(sorted(set(signal_formats)))
        raise NotImplementedError(f'{signal_format_string} not implemented')

    import wfdb
    wfdb.wrsamp(record, fs=sampling_frequency, units=signal_units, sig_name=signal_names, \
                d_signal=signal, fmt=signal_formats, adc_gain=adc_gains, baseline=baselines, comments=comments, \
                write_dir=path)

# Save the signal(s) for a record.
def save_signals(record, signals):
    save_signal(record, signals)

# Save the dx class(es) for a record.
def save_dx(record, dx):
    header_file = get_header_file(record)
    header = load_text(header_file)
    header += '#Dx: ' + ', '.join(dx) + '\n'
    save_text(header_file, header)
    return header

def save_dxs(record, dxs):
    return save_dx(record, dxs)

### Helper Challenge functions

# Load a text file as a string.
def load_text(filename):
    with open(filename, 'r') as f:
        string = f.read()
    return string

# Save a string as a text file.
def save_text(filename, string):
    with open(filename, 'w') as f:
        f.write(string)

# Get a variable from a string.
def get_variable(string, variable_name):
    variable = ''
    has_variable = False
    for l in string.split('\n'):
        if l.startswith(variable_name):
            variable = l[len(variable_name):].strip()
            has_variable = True
    return variable, has_variable

# Get variables from a string.
def get_variables(string, variable_name, sep=','):
    variables = list()
    has_variable = False
    for l in string.split('\n'):
        if l.startswith(variable_name):
            variables += [variable.strip() for variable in l[len(variable_name):].strip().split(sep)]
            has_variable = True
    return variables, has_variable

# Get the signal file(s) from a header or a similar string.
def get_signal_files_from_header(string):
    signal_files = list()
    for i, l in enumerate(string.split('\n')):
        arrs = [arr.strip() for arr in l.split(' ')]
        if i==0 and not l.startswith('#'):
            num_channels = int(arrs[1])
        elif i<=num_channels and not l.startswith('#'):
            signal_file = arrs[0]
            if signal_file not in signal_files:
                signal_files.append(signal_file)
        else:
            break
    return signal_files

# Get the image file(s) from a header or a similar string.
def get_image_files_from_header(string):
    images, has_image = get_variables(string, '#Image:')
    if not has_image:
        raise Exception('No images available: did you forget to generate or include the images?')
    return images

# Get the dx class(es) from a header or a similar string.
def get_dxs_from_header(string):
    dxs, has_dx = get_variables(string, '#Dx:') #NHATnote: take the word "Normal" or "Abnormal" out from .hea file, prepared before by the function provided in the code, which read the .csv files to put in Normal or Abnormal
    if not has_dx:
        raise Exception('No dx classes available: are you trying to load the classes from the held-out dataset, or did you forget to prepare the data to include the classes?')
    return dxs

# Get the header file for a record.
def get_header_file(record):
    if not record.endswith('.hea'):
        header_file = record + '.hea'
    else:
        header_file = record
    return header_file

# Get the signal file(s) for a record.
def get_signal_files(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    signal_files = get_signal_files_from_header(header)
    return signal_files

# Get the image file(s) for a record.
def get_image_files(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    image_files = get_image_files_from_header(header)
    return image_files

### WFDB functions

# Get the record name from a header file.
def get_record_name(string):
    value = string.split('\n')[0].split(' ')[0].split('/')[0].strip()
    return value

# Get the number of signals from a header file.
def get_num_signals(string):
    value = string.split('\n')[0].split(' ')[1].strip()
    if is_integer(value):
        value = int(value)
    else:
        value = None
    return value

# Get the sampling frequency from a header file.
def get_sampling_frequency(string):
    value = string.split('\n')[0].split(' ')[2].split('/')[0].strip()
    if is_number(value):
        value = float(value)
    else:
        value = None
    return value

# Get the number of samples from a header file.
def get_num_samples(string):
    value = string.split('\n')[0].split(' ')[3].strip()
    if is_integer(value):
        value = int(value)
    else:
        value = None
    return value

# Get the signal formats from a header file.
def get_signal_formats(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[1]
            if 'x' in field:
                field = field.split('x')[0]
            if ':' in field:
                field = field.split(':')[0]
            if '+' in field:
                field = field.split('+')[0]
            value = field
            values.append(value)
    return values

# Get the ADC gains from a header file.
def get_adc_gains(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                field = field.split('/')[0]
            if '(' in field and ')' in field:
                field = field.split('(')[0]
            value = float(field)
            values.append(value)
    return values

# Get the baselines from a header file.
def get_baselines(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                field = field.split('/')[0]
            if '(' in field and ')' in field:
                field = field.split('(')[1].split(')')[0]
            else:
                field = get_adc_zeros(string)[i-1]
            value = int(field)
            values.append(value)
    return values

# Get the signal units from a header file.
def get_signal_units(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                value = field.split('/')[1]
            else:
                value = 'mV'
            values.append(value)
    return values

# Get the ADC resolutions from a header file.
def get_adc_resolutions(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[3]
            value = int(field)
            values.append(value)
    return values

# Get the ADC zeros from a header file.
def get_adc_zeros(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[4]
            value = int(field)
            values.append(value)
    return values

# Get the initial values of a signal from a header file.
def get_initial_values(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[5]
            value = int(field)
            values.append(value)
    return values

# Get the checksums of a signal from a header file.
def get_checksums(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[6]
            value = int(field)
            values.append(value)
    return values

# Get the block sizes of a signal from a header file.
def get_block_sizes(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[7]
            value = int(field)
            values.append(value)
    return values

# Get the signal names from a header file.
def get_signal_names(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            value = l.split(' ')[8]
            values.append(value)
    return values

### Evaluation functions

# Construct the binary one-vs-rest confusion matrices, where the columns are the expert labels and the rows are the classifier
# for the given classes.
def compute_one_vs_rest_confusion_matrix(labels, outputs, classes):
    assert np.shape(labels) == np.shape(outputs)

    num_instances = len(labels)
    num_classes = len(classes)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_instances):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1: # TP
                A[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1: # FP
                A[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0: # FN
                A[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0: # TN
                A[j, 1, 1] += 1

    return A

# Compute macro F-measure.



# NHATedit compute accuracy
def compute_accuracy(labels, outputs):
    # Compute confusion matrix.
    #classes = sorted(set.union(*map(set, labels)))
    classes = ['Normal', 'Abnormal']
    print("classes in compute_accuracy:", classes)
    
    x= 0
    for i in range(len(labels)):
        if labels[i]==outputs[i] :
            x+=1
    print(x,x,x)
    #print(labels[0],"1")
    
    labels = compute_one_hot_encoding(labels, classes)
    outputs = compute_one_hot_encoding(outputs, classes)
    #print(labels[0],"1")
    

    #raise(Exception("testthis"))
    A = compute_one_vs_rest_confusion_matrix(labels, outputs, classes)
    #print(A)
    

    num_classes = len(classes)
    per_class_accuracy = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        #print(tp,fp,fn,tn)
        #raise Exception("Tetsts")
        if 2 * tp + fp + fn > 0:
            per_class_accuracy[k] = float(tp) / float(tp + tn + fp + fn) #Cho nay bi sai perclass nhung ma all_acc van dung
        else:
            per_class_accuracy[k] = float('nan')

    if np.any(np.isfinite(per_class_accuracy)):
        all_accuracy = np.nansum(per_class_accuracy)
    else:
        mean_accuracy = float('nan')

    return all_accuracy, per_class_accuracy, classes


def compute_f_measure(labels, outputs):
    # Compute confusion matrix.
    # classes = sorted(set.union(*map(set, labels)))
    classes = ['Normal', 'Abnormal']
    print("classes in compute_f_measure:", classes)
    labels = compute_one_hot_encoding(labels, classes)
    outputs = compute_one_hot_encoding(outputs, classes)
    A = compute_one_vs_rest_confusion_matrix(labels, outputs, classes)

    num_classes = len(classes)
    per_class_f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            per_class_f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            per_class_f_measure[k] = float('nan')

    if np.any(np.isfinite(per_class_f_measure)):
        macro_f_measure = np.nanmean(per_class_f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, per_class_f_measure, classes

# Reorder channels in signal.
def reorder_signal(input_signal, input_channels, output_channels):
    # Do not allow repeated channels with potentially different values in a signal.
    assert(len(set(input_channels)) == len(input_channels))
    assert(len(set(output_channels)) == len(output_channels))

    if input_channels == output_channels:
        output_signal = input_signal
    else:
        input_channels = [channel.strip().casefold() for channel in input_channels]
        output_channels = [channel.strip().casefold() for channel in output_channels]

        input_signal = np.asarray(input_signal)
        num_samples = np.shape(input_signal)[0]
        num_channels = len(output_channels)
        data_type = input_signal.dtype
        output_signal = np.zeros((num_samples, num_channels), dtype=data_type)

        for i, output_channel in enumerate(output_channels):
            for j, input_channel in enumerate(input_channels):
                if input_channel == output_channel:
                    output_signal[:, i] = input_signal[:, j]

    return output_signal

# Pad or truncate signal.
def trim_signal(input_signal, num_samples_trimmed):
    input_signal = np.asarray(input_signal)
    num_samples, num_channels = np.shape(input_signal)
    data_type = input_signal.dtype

    if num_samples == num_samples_trimmed:
        output_signal = input_signal
    else:
        output_signal = np.zeros((num_samples_trimmed, num_channels), dtype=data_type)
        if num_samples < num_samples_trimmed: # Zero-pad the signals.
            output_signal[:num_samples, :] = input_signal
        else: # Truncate the signals.
            output_signal = input_signal[:num_samples_trimmed, :]

    return output_signal


# Compute a metric inspired by the Kolmogorov-Smirnov test statistic.
def compute_ks_metric(label_signal, output_signal):
    label_signal = np.asarray(label_signal)
    output_signal = np.asarray(output_signal)

    assert(label_signal.ndim == output_signal.ndim == 1)
    assert(np.size(label_signal) == np.size(output_signal))

    idx_finite_signal = np.isfinite(label_signal)
    label_signal = label_signal[idx_finite_signal]
    output_signal = output_signal[idx_finite_signal]

    idx_nan_signal = np.isnan(output_signal)
    output_signal[idx_nan_signal] = 0

    label_signal_cdf = np.cumsum(np.abs(label_signal))
    output_signal_cdf = np.cumsum(np.abs(output_signal))

    if label_signal_cdf[-1] > 0:
        label_signal_cdf = label_signal_cdf / label_signal_cdf[-1]
    if output_signal_cdf[-1] > 0:
        output_signal_cdf = output_signal_cdf / output_signal_cdf[-1]

    goodness_of_fit = 1.0 - np.max(np.abs(label_signal_cdf - output_signal_cdf))

    return goodness_of_fit

# Compute the adaptive signed correlation index (ASCI) metric.
def compute_asci_metric(label_signal, output_signal, beta=0.05):
    label_signal = np.asarray(label_signal)
    output_signal = np.asarray(output_signal)

    assert(label_signal.ndim == output_signal.ndim == 1)
    assert(np.size(label_signal) == np.size(output_signal))

    idx_finite_signal = np.isfinite(label_signal)
    label_signal = label_signal[idx_finite_signal]
    output_signal = output_signal[idx_finite_signal]

    idx_nan_signal = np.isnan(output_signal)
    output_signal[idx_nan_signal] = 0

    if beta <= 0 or beta > 1:
        raise ValueError('The beta value should be greater than 0 and less than or equal to 1.')

    threshold = beta * np.std(label_signal)

    noise_signal = np.abs(label_signal - output_signal)

    discrete_noise = np.zeros_like(noise_signal)
    discrete_noise[noise_signal <= threshold] = 1
    discrete_noise[noise_signal > threshold] = -1

    asci = np.mean(discrete_noise)

    return asci

# Compute a weighted absolute difference metric.
def compute_weighted_absolute_difference(label_signal, output_signal, sampling_frequency):
    label_signal = np.asarray(label_signal)
    output_signal = np.asarray(output_signal)

    assert(label_signal.ndim == output_signal.ndim == 1)
    assert(np.size(label_signal) == np.size(output_signal))

    idx_finite_signal = np.isfinite(label_signal)
    label_signal = label_signal[idx_finite_signal]
    output_signal = output_signal[idx_finite_signal]

    idx_nan_signal = np.isnan(output_signal)
    output_signal[idx_nan_signal] = 0

    from scipy.signal import filtfilt

    m = round(0.1 * sampling_frequency)
    w = filtfilt(np.ones(m), m, label_signal, method='gust')
    w = 1 - 0.5/np.max(w) * w
    n = np.sum(w)

    weighted_absolute_difference_metric = np.sum(np.abs(label_signal-output_signal) * w)/n

    return weighted_absolute_difference_metric

### Other helper functions

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Check if a variable is a NaN (not a number) or represents a NaN.
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False

# Cast a value to an integer if an integer, a float if a non-integer float, and an unknown value otherwise.
def cast_int_float_unknown(x):
    if is_integer(x):
        x = int(x)
    elif is_finite_number(x):
        x = float(x)
    elif is_number(x):
        x = 'Unknown'
    else:
        raise NotImplementedError(f'Unable to cast {x}.')
    return x

# Construct the one-hot encoding of data for the given classes.
def compute_one_hot_encoding(data, classes):
    num_instances = len(data)
    num_classes = len(classes)

    one_hot_encoding = np.zeros((num_instances, num_classes), dtype=np.bool_)
    unencoded_data = list()
    for i, x in enumerate(data):
        for y in x:
            for j, z in enumerate(classes):
                if (y == z) or (is_nan(y) and is_nan(z)):
                    one_hot_encoding[i, j] = 1

    return one_hot_encoding