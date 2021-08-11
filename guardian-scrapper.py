import time
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen as uReq
import json
import pandas as pd

cookies = dict(name='jerry', password='888')

URL = 'https://www.theguardian.com/world/coronavirus-outbreak/all'
page = requests.get(URL, cookies = cookies)

soup = BeautifulSoup(page.content, 'html.parser')

links = []
titles = []

base = []
URL = 'https://www.theguardian.com/world/coronavirus-outbreak/all'
page = requests.get(URL, cookies = cookies)
soup = BeautifulSoup(page.content, 'html.parser')
containers3 = ['a']

i = 2
linkar = []
while len(containers3) != 0 and len(linkar) < 11761:
        URL = 'https://www.theguardian.com/world/coronavirus-outbreak?page=' + str(i)
        page = requests.get(URL, cookies = cookies)
        soup = BeautifulSoup(page.content, 'html.parser')
        containers3 = soup.findAll("div", {"class": "fc-item__container"})
        for j in containers3:
            zz = j.find("a", {"class": "fc-item__link"})
            linkar.append(zz.text)
        i = i+1
        
        
#662 seconds (11 minutes)
linkar = pd.DataFrame(linkar)
linkar['Label'] = 'TRUE'
linkar = linkar.rename(index={'0': "Title"})

linkar.columns = ['Title', 'Label']


linkar.to_csv("Guardian.csv")
