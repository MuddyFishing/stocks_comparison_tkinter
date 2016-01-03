# stocks-comparison
A stock comparison app built on Python Tkinter package

Project proposes to compare stocks based on their historical data and try to use prediction models to approximate their future performance.  See images __Demo 1.jpg__ and __Demo 2.jpg__ in the root for more details.

1. It is very simple GUI based app where user may add multiple tickers and give a date range.    
2. App has few listed FE101 operations like:    
    Moving average  
  	Returns, logged returns  
  	Variance, logged variance, and plot them against each other to assess how they did comparatively.  
  	Tried fitting them with predictive models like:  
  	ARIMA  
  	ACF/ PACF  
  	Autocorrelations  
  	Regressions, etc,  
  and we could chose to build any/ all of these plots for the stocks.    
3. In a seprate widget it shows current stock data.    
4. A secondary widget further compares stocks with the SPY500 (i.e. market) for the last 5 years.

Future:  
1. Add more models for predictions.  
2. Improve GUI.  
3. Implement Google analytics API to find correlation between trends to improve predictions. 

Credits:  
1. http://gouthamanbalaraman.com/blog/calculating-stock-beta.html  
2. http://www.johnwittenauer.net/a-simple-time-series-analysis-of-the-sp-500-index/  
3. http://francescopochetti.com/category/stock-project/  
4. Moshe from SO and his resource@ https://github.com/moshekaplan/tkinter_components/tree/master/CalendarDialog  

I have given credits in all the code pieces where they were due and in my reports/ presentations that I'd made for my coursework. Please let me know if there're any inadequecies on my part.

@Author:   
Sandeep Joshi  
09/12/2015
