
#%% From https://pythonprogramminglanguage.com/get-links-from-webpage/
## Download all links to a list
import wget
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

req = Request("http://biggeo.gvc.gu.se/Temp/tracking/GPM/")
html_page = urlopen(req)

soup = BeautifulSoup(html_page, "lxml")

links = []
for link in soup.findAll('a'):
    links.append(link.get('href'))
    
#%% Downloading mask files between may and sept
base_url = 'http://biggeo.gvc.gu.se/Temp/tracking/GPM/'
path = 'D:/GPM_masks/'
years = ['2001','2002','2003','2004', '2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']
for y in years:
    s_idx = links.index('Mask_segmentation_'+y+'05.nc')
    e_idx = links.index('Mask_segmentation_'+y+'09.nc')
    y_links = links[s_idx:e_idx+1]
    for l in y_links:
        filename = wget.download(base_url + l, out=path)
        
#%% Downloading precipitation files for one year
start = '3B-HHR.MS.MRG.3IMERG.20090501-S000000-E002959.0000.V06B.HDF5.nc4'
end = '3B-HHR.MS.MRG.3IMERG.20090930-S233000-E235959.1410.V06B.HDF5.nc4'
st_idx = links.index(start); e_idx = links.index(end)
links_may_sept = links[st_idx:e_idx+1]
del links, start, end, st_idx, e_idx, req, html_page

## Change year between downloads
base_url = 'http://biggeo.gvc.gu.se/Temp/GPM/Y2009/'
path = 'D:/GPM_prec/2009' 
for l in links_may_sept:
    filename = wget.download(base_url + l, out=path)
    
#%% Downloading all GPM precipitation files between may and september from biggeo.gvc.gu.se for each year
path = 'D:/GPM_prec/'
years = ['2001','2002','2003','2004', '2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016']
links = []
for y in years:
    req = Request('http://biggeo.gvc.gu.se/Temp/GPM/Y'+y+'/')
    html_page = urlopen(req)
    soup = BeautifulSoup(html_page, "lxml")
    for link in soup.findAll('a'):
        links.append(link.get('href'))
    start = '3B-HHR.MS.MRG.3IMERG.'+y+'0501-S000000-E002959.0000.V06B.HDF5.nc4'
    end = '3B-HHR.MS.MRG.3IMERG.'+y+'0930-S233000-E235959.1410.V06B.HDF5.nc4'
    st_idx = links.index(start); 
    e_idx = links.index(end)
    links_may_sept = links[st_idx:e_idx+1]
    url = 'http://biggeo.gvc.gu.se/Temp/GPM/Y'+y+'/'
    for l in links_may_sept:
            filename = wget.download(url + l, out=path)