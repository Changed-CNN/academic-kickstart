+++
title= "Applet: real-time stock price dynamic chart"
date=2019-09-06
tags=["Stock", "Python", "Matplotlib", "Tushare", "Applet", "Real-time"]

+++

## Goal

The design of this program allows us to monitor the stock price in real time, and in the form of a dynamic graph, when the price rises in red line segment, when the price is unchanged in yellow line segment, when the price falls in green line segment, and users can customize the effect.

## Problems&Solutions

- **How to get real-time prices and convert data types？**

  **`Tushare`** is a free, open source python financial data interface package. It mainly realizes the process of collecting, cleaning, processing and storing financial data such as stocks, and can provide fast, clean and diverse data for easy analysis. Most of the data format returned by Tushare is of type pandas DataFrame, so we also need to turn data of type `DataFrame` into data of type `Float` for the sake of graphing.

  > *Tips: The price of stock only change on weekdays, so the effect only can be realized on weekdays.*

  ```python
  def get_price(STOCK):
      df = ts.get_realtime_quotes(STOCK)
      price=float(list(df.price)[0])
      return price
  ```

- **How to make a dynamic graph？**

  **`Time`** is often used here, because we need to get the price every once in a while, the function **`sleep`** is applied to execute a loop every few seconds.

  ```python
  time_remaining = interval-time.time()%interval
  time.sleep(time_remaining)
  ```

  To record the price at each moment and optimize the display of the time axis, three lists are used to record `float time`, `float price`, and `string time` respectively.

  ```python
  ntime=time.time()
  atime.append(ntime)
  nprice = get_price(STOCK)
  aprice.append(nprice)
  nc_time=time.strftime("%H:%M:%S", time.localtime())
  c_time.append(nc_time)
  ```

  **`Matplotlib`** was used to draw the graph. Coordinate axes were set and lines were drawn. Different changes were represented by different colored lines.

  ```python
  plt.xlabel("time")
  plt.ylabel("price")
  plt.title("Dynamic price of stock")
  plt.xticks(atime, c_time, fontsize=7, rotation=60)
  x = [atime[i],atime[i+1]]
  y = [aprice[i],aprice[i+1]]
  if aprice[i]<aprice[i+1]:
      plt.plot(x,y,color='r')
      plt.scatter(x,y,color='r')
  elif aprice[i]==aprice[i+1]:
      plt.plot(x,y,color='y')
      plt.scatter(x, y,color='y')
  else :
      plt.plot(x,y,color='g')
      plt.scatter(x, y,color='g')
  ```

- **How to let the users customize？**

  We provide three parameters that can be customized by the users:

  1. **interval**: number of seconds between refreshes
  2. **STOCK**: stock symbol
  3. **T**: refresh times

  ```python
  interval = 5
  STOCK = ['000001']
  T=30
  ```

## Experimental Effect

![price](/img/price.png)

## Complete Code

```python
import time
import matplotlib.pyplot as plt
import tushare as ts
import pandas as pd
def print_ts(message):
    print ("[%s] %s"%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message))

def get_price(STOCK):
    df = ts.get_realtime_quotes(STOCK)
    price=float(list(df.price)[0])
    return price

def run(interval):
    print_ts("-"*100)
    print_ts("Starting every %s seconds."%interval)
    print_ts("-"*100)
    t=0
    atime=[]
    aprice=[]
    c_time=[]

    while t<T:
        t=t+1
        time_remaining = interval-time.time()%interval
        time.sleep(time_remaining)
        print_ts("Updating at %s"%(time.ctime()))
        nc_time=time.strftime("%H:%M:%S", time.localtime())
        c_time.append(nc_time)
        ntime=time.time()
        atime.append(ntime)
        nprice = get_price(STOCK)
        print(nprice)
        aprice.append(nprice)
        for i in range(0,len(aprice)-2):
            plt.xlabel("time")
            plt.ylabel("price")
            plt.title("Dynamic price of stock")
            plt.xticks(atime, c_time, fontsize=7, rotation=60)
            x = [atime[i],atime[i+1]]
            y = [aprice[i],aprice[i+1]]
            if aprice[i]<aprice[i+1]:
                plt.plot(x,y,color='r')
                plt.scatter(x,y,color='r')
            elif aprice[i]==aprice[i+1]:
                plt.plot(x,y,color='y')
                plt.scatter(x, y,color='y')
            else :
                plt.plot(x,y,color='g')
                plt.scatter(x, y,color='g')
        plt.show()

if __name__=="__main__":
    interval = 5
    STOCK = ['000001']
    T=30
    run(interval)
```

