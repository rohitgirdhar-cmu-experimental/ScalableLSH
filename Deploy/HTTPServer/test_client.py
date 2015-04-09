#!/usr/bin/env python
import sys
import urllib2
import urllib

from threading import Thread

def make_request(url):
    data = "http://10.3.2.61/~ubuntu/projects/001_backpage/dataset/corpus/ImagesTexas/Texas_2012_10_10_1349853753000_5_1.jpg"
    headers = { 'Content-type' : 'text',  'Content-length' : str(len(data))}
    req = urllib2.Request(url, data, headers) #POST request
    try:
      response = urllib2.urlopen(req)
      result = response.read()
      print result
    except urllib2.URLError, err:
      print err

def main():
    port = 8888
    try:
      make_request("http://10.1.94.128:%d" % port)
    except urllib2.HTTPError, err:
      print err

main()
