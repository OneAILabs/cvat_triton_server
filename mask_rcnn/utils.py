def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    if class_names[0] != 'BG':
        class_names.insert(0, 'BG')
    return class_names
