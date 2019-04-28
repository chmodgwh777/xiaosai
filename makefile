all: data runPython

data: rawData
	awk -f test.awk rawData > data

runPython: data
	python main.py