
#! /usr/bin/python3
'''
MLB 23 Best Buys Calculator
https://mlb24.theshow.com/apis/listings.json?page=1
Authors: Brett Tubb
'''
import urllib.request, json
# from replit import db
import threading
import time
import os

tax = .9
margin = 5
results = []
maxmoney = 2000
minimums = {"Diamond": 5000, "Gold": 1000, "Silver": 100, "Bronze": 20, "Common": 5}

series = 1337 # 1337 if Live players or blank for all
num_results = 10 # How many results to give
api_url = "https://mlb23.theshow.com/apis/listings.json"

def getNumPages(api_url: str, series: int):
  with urllib.request.urlopen(f"{api_url}?series_id={series}") as url:
    data = json.loads(url.read().decode())
    print(f"Total page count {data['total_pages']}", flush=True)
    return data['total_pages']

def getPage(api_url: str, num: int, series: int):
  # print(f"Getting page {num}")
  with urllib.request.urlopen(f"{api_url}?page={num}&series_id={series}") as url:
   checkProfit(json.loads(url.read().decode()))
    # print(f" {n}", end="", flush=True)
    # print("\n\nWe're done!! Doing some crazy math...\n")

# def storePage(items: dict):
#   for listing in items['listings']:
#     db['listing_name'] = listing

# def listAll():
#   for key in db.keys():
#     print(key)
#   return db.keys()

def checkProfit(data: dict):
  '''Processes data and appends results if profit over margin. '''
  for listing in data['listings']:
    name = listing['listing_name']
    sell = listing['best_buy_price']
    buy = listing['best_sell_price']
    rarity = listing['item']['rarity']
    uuid = listing['item']['uuid']
    sell = max(minimums[rarity], sell) # make sure sell is minimum for rarity
    profit = int(buy*tax-sell)
    if profit > margin and buy < maxmoney:
      results.append([profit, name, buy, uuid])

def printResults(results):
  print('\n\n')
  print(f"{'#': <3}  {'Profit': >7} {'Name': >24}    {'Price': >8}")
  if len(results) >= num_results:
    for i in range(num_results):
      print(f"{i + 1: <3} ${results[i][0]: >7,} {results[i][1]: >24}   ${results[i][2]: >8,}")
      print(f"https://mlb23.theshow.com/items/{results[i][3]}\n")
  else:
    for i, n in enumerate(results):  
      print(f"{i + 1: <3} ${results[i][0]: >7,} {results[i][1]: >24}   ${results[i][2]: >8,}")
      print(f"https://mlb23.theshow.com/items/{results[i][3]}\n")

print("We are live. Connecting to the mothership... Searching for sprockets.\n")      
# Run the script
while True:
  print("Time to do some work... \n")
  # for n in range(1, getNumPages(api_url, series), 6):
  try:
    print('Starting downloads... ', flush=True)
    threads = []
    for n in range(1, getNumPages(api_url, series)):
      t = threading.Thread(target=getPage, args=(api_url, n, series))
      threads.append(t)
      t.start()
  except:
    print("Error: unable to start threads")

  time.sleep(2)
  results.sort(reverse=True)
  os.system('clear') # Clear console before printing new results, keeps location static
  printResults(results)
  results = [] # Clear results before beginning next pull.
  time.sleep(10) # Sleep 10 seconds before refresh
