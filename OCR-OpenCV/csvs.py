import datetime 
import csv
dt=datetime.datetime.now()
datef=dt.strftime("%x")
timef=dt.strftime("%X")
text = input('Enter Something text: ')

with open('ofaj_data.csv', 'w') as f:
    w = csv.writer(f, quoting=csv.QUOTE_ALL) 
    w.writerow([datef , timef, text])

# import csv
# with open('file.csv', 'w') as f:
#     w = csv.writer(f, quoting=csv.QUOTE_ALL) 

#     while (1):
#         why = input("why? ")
#         date = input("date: ")
#         story = input("story: ")
#         w.writerow([why, date, story])