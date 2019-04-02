
In this example, we will extract a table of monthly and annual rainfall from 
8 Northern California stations. The data is made available by the [state department of water resources](http://cdec.water.ca.gov/cgi-progs/precip1/8STATIONHIST').

On inspecting the source of the page, the data is contained near lines 475-588. Here, we'll extract it and save it in a file.


```python
import urllib.request
webdata = urllib.request.urlopen('http://cdec.water.ca.gov/cgi-progs/precip1/8STATIONHIST').read()
```


```python
data = str(webdata).split('\\n')[481:600]
data[:10]
```


```python
data[-2]
```

The data contains some blank lines. We will remove these and split the lines into separate elements:


```python
data = [l.split() for l in data if l]
```

Now let's write this to a file.


```python
import csv
outfile = open('rainfall.csv', 'w')
writer = csv.writer(outfile)
for row in data:
    writer.writerow(row)
outfile.close()
```


```python

```
