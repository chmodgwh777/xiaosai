data = [];

def readData(s):
    return [float(f) for f in s.split()]

currentLine = ''
with open('/Users/gao/Desktop/xiaosai/data', 'r') as f:
    currentLine = f.readline()
    while (currentLine):
        if currentLine[0].isdigit():
            data.append(readData(currentLine))
        currentLine = f.readline()
